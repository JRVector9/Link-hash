from decimal import Decimal
from django.db import transaction, models
from django.utils import timezone
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

from market.models import Game, GameSession, Participation, Winner
import logging
from celery import current_app

log = logging.getLogger(__name__)

def _format_hms(secs: int) -> str:
    secs = max(0, int(secs))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _compute_required_bet_sol(game: Game, session: GameSession) -> Decimal:
    first_floor = Decimal("0.01")
    minb = game.min_bet_sol or Decimal("0")

    # ✅ ONLY source of truth for mode:
    is_limit = (game.subtype == "limit")

    # ✅ Ignore admin increment_pct forever:
    inc = Decimal("0") if is_limit else Decimal("3")

    # In limit mode, fixed entry is used; in no-limit, fixed is ignored even if DB has a value.
    fixed = (game.fixed_bet_sol or Decimal("0")) if is_limit else Decimal("0")

    last_verified = (
        Participation.objects.filter(session=session, status=Participation.STATUS_VERIFIED)
        .order_by("-bet_index", "-created_at")
        .first()
    )

    # ✅ Limit: fixed entry
    if is_limit:
        return max(first_floor, fixed).quantize(Decimal("0.000000001"))

    # ✅ No-limit: first bet is floor/min
    if not last_verified:
        return max(first_floor, minb).quantize(Decimal("0.000000001"))

    prev_amt = last_verified.amount_sol or Decimal("0")
    next_amt = (prev_amt * (Decimal("1") + (inc / Decimal("100")))).quantize(Decimal("0.000000001"))
    return max(first_floor, minb, next_amt).quantize(Decimal("0.000000001"))

def _get_king_participation(game: Game, session: GameSession):
    """
    Returns the 'king' (last/winning bettor) for a session.
    - limit mode: ordered by block_time (on-chain truth)
    - no_limit mode: ordered by bet_index (submission order)
    """
    qs = (
        Participation.objects.filter(session=session, status=Participation.STATUS_VERIFIED)
        .select_related("user_profile")
    )
    if game.subtype == "limit":
        return qs.order_by(
            models.F("block_time").desc(nulls_last=True),
            "-created_at",
        ).first()
    return qs.order_by("-bet_index", "-created_at").first()

def _broadcast_game_update(game: Game, session: GameSession) -> None:
    king_part = _get_king_participation(game, session)
    king_wallet = king_part.user_profile.wallet_address if king_part else "Unknown"

    now = timezone.now()
    ends_at = session.fomo_ends_at
    timer_secs = int(max(0, (ends_at - now).total_seconds())) if ends_at else 0

    required = _compute_required_bet_sol(game, session)

    prev_bid_sol = str(king_part.amount_sol) if king_part and king_part.amount_sol is not None else "0"

    bets_count = Participation.objects.filter(session=session, status=Participation.STATUS_VERIFIED).count()

    is_limit = (game.subtype == "limit")
    inc = Decimal("0") if is_limit else Decimal("3")


    payload = {
        "type": "game_update",
        "game": {
            "id": game.id,
            "name": game.name,
            "category": game.category,
            "subtype": game.subtype,
            "protocol_fee_pct": str(game.protocol_fee_pct),
            "sol_wallet": game.sol_public_key,
            "min_bet_sol": str(game.min_bet_sol),
            "fixed_bet_sol": str(game.fixed_bet_sol),
            "increment_pct": str(inc),
        },
        "session": {
            "id": session.id,
            "round_no": session.round_no,
            "status": session.status,
            "pot_total_sol": str(session.pot_total_sol),
            "king_wallet": king_wallet,
            "fomo_ends_at": ends_at.isoformat() if ends_at else None,
            "timer": _format_hms(timer_secs),
            "required_bet_sol": str(required),
            "prev_bid_sol": prev_bid_sol,
            "bets_count": bets_count,
        }
    }

    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f"game_{game.id}",
        {"type": "game_update", "payload": payload},
    )


def _broadcast_past_result(game: Game, settled_session: GameSession) -> None:
    """
    Broadcast ONE new past-result row for this game (when a session settles).
    Skips sessions with no winner.
    """
    win = (
        Winner.objects.filter(session=settled_session)
        .select_related("user_profile", "participation")
        .order_by("-created_at")
        .first()
    )
    if not (win and win.user_profile):
        return

    winner_wallet = win.user_profile.wallet_address or "Unknown"
    if winner_wallet != "Unknown" and len(winner_wallet) > 12:
        winner_disp = f"{winner_wallet[:4]}...{winner_wallet[-4:]}"
    else:
        winner_disp = winner_wallet

    row = {
        "winner": winner_disp,
        "bets": Participation.objects.filter(session=settled_session, status=Participation.STATUS_VERIFIED).count(),
        "last_bet": float(win.participation.amount_sol) if win.participation else 0.0,
        "pot": float(settled_session.pot_total_sol or Decimal("0")),
        "ended": settled_session.settled_at.strftime("%Y-%m-%d %H:%M:%S") if settled_session.settled_at else "—",
        "game_id": game.id,
        "subtype": game.subtype,
    }

    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f"game_{game.id}",
        {"type": "past_result", "payload": {"type": "past_result", "row": row}},
    )


def _schedule_fomo_settlement(session: GameSession) -> None:
    """
    Schedule settle_fomo_session reliably.

    ✅ Uses countdown instead of eta to avoid timezone/UTC interpretation issues.
    ✅ If scheduled early/late, the task itself will self-heal (see tasks.py changes below).
    """
    if not session.fomo_ends_at:
        return

    # revoke old if exists
    if session.settle_task_id:
        try:
            current_app.control.revoke(session.settle_task_id, terminate=False)
        except Exception as e:
            log.warning("Failed to revoke old settle task %s: %s", session.settle_task_id, e)

    try:
        now = timezone.now()
        delay = (session.fomo_ends_at - now).total_seconds()
        if delay < 0:
            delay = 0

        result = current_app.send_task(
            "market.tasks.settle_fomo_session",
            args=[session.id],
            countdown=delay,
        )
        session.settle_task_id = result.id
        session.save(update_fields=["settle_task_id"])
        log.info(
            "Scheduled settle_fomo_session session=%s ends_at=%s delay=%.3fs task=%s",
            session.id, session.fomo_ends_at, delay, result.id
        )
    except Exception as e:
        log.exception("FAILED to schedule settlement for session=%s: %s", session.id, e)


def _start_new_fomo_session(game: Game) -> GameSession:
    """
    Create a new OPEN session with pot=0 and fomo_ends_at=now+timer,
    AND schedule its Celery settlement immediately.
    """
    last = GameSession.objects.filter(game=game).order_by("-round_no").first()
    next_round = (last.round_no + 1) if last else 1

    now = timezone.now()
    ends_at = now + timezone.timedelta(seconds=int(game.fomo_timer_seconds or 600))

    session = GameSession.objects.create(
        game=game,
        round_no=next_round,
        status=GameSession.STATUS_OPEN,
        pot_total_sol=Decimal("0"),
        fee_pct_snapshot=game.protocol_fee_pct,
        fomo_ends_at=ends_at,
    )

    _schedule_fomo_settlement(session)
    return session



def _settle_expired_fomo_session_if_needed(game: Game, session: GameSession) -> GameSession:
    """
    On-demand settlement (used by page/API fetch).
    Celery-based settlement is the real production path, but this stays as a backup.
    """
    if session.status != GameSession.STATUS_OPEN:
        return _start_new_fomo_session(game)

    if not session.fomo_ends_at:
        session.fomo_ends_at = timezone.now() + timezone.timedelta(seconds=int(game.fomo_timer_seconds or 600))
        session.save(update_fields=["fomo_ends_at"])
        return session

    now = timezone.now()
    if session.fomo_ends_at > now:
        return session

    last_verified = _get_king_participation(game, session)

    session.status = GameSession.STATUS_LOCKED
    session.save(update_fields=["status"])

    if not last_verified:
        session.status = GameSession.STATUS_SETTLED
        session.settled_at = now
        session.save(update_fields=["status", "settled_at"])
        new_sess = _start_new_fomo_session(game)
        _broadcast_game_update(game, new_sess)
        return new_sess

    gross = session.pot_total_sol or Decimal("0")
    fee_pct = session.fee_pct_snapshot or Decimal("0")
    fee_amt = (gross * fee_pct / Decimal("100")).quantize(Decimal("0.000000001"))
    payout = (gross - fee_amt).quantize(Decimal("0.000000001"))
    if payout < 0:
        payout = Decimal("0")

    Winner.objects.create(
        session=session,
        user_profile=last_verified.user_profile,
        participation=last_verified,
        payout_amount_sol=payout,
        payout_status=Winner.PAYOUT_PENDING,
    )

    session.status = GameSession.STATUS_SETTLED
    session.settled_at = now
    session.save(update_fields=["status", "settled_at"])

    _broadcast_past_result(game, session)  # ✅ winner-only

    new_sess = _start_new_fomo_session(game)
    _broadcast_game_update(game, new_sess)
    return new_sess



def _ensure_current_fomo_session(game: Game) -> GameSession:
    """
    Guarantees there is a CURRENT OPEN session for this game.

    ✅ Critical: ensures the OPEN session has a scheduled Celery settlement,
    even if nobody bets (so no-winner sessions still advance automatically).
    """
    session = (
        GameSession.objects.filter(game=game, status=GameSession.STATUS_OPEN)
        .order_by("-created_at")
        .first()
    )

    if not session:
        session = _start_new_fomo_session(game)  # this already schedules settlement
        _broadcast_game_update(game, session)
        return session

    # ✅ Self-heal: if an OPEN session exists but has no scheduled settlement,
    # schedule it now (covers sessions created by old code paths).
    if not session.settle_task_id:
        _schedule_fomo_settlement(session)

    # resolve expiry if needed (backup path)
    session = _settle_expired_fomo_session_if_needed(game, session)

    # ✅ after settling, we might be holding a NEW open session; ensure it’s scheduled too
    if session.status == GameSession.STATUS_OPEN and not session.settle_task_id:
        _schedule_fomo_settlement(session)

    return session

