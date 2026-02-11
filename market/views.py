# market/views.py
import json
import secrets
import requests

from decimal import Decimal, InvalidOperation
from django.conf import settings
from django.contrib import messages
from django.core.paginator import Paginator
from django.db import IntegrityError, transaction, models
from django.db.models import Case, When, Value, IntegerField
from django.db.models.functions import Coalesce
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.utils import timezone
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import (
    require_GET,
    require_POST,
    require_http_methods,
)
from django.core.cache import cache

from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError

from django.db.models import F
from django.core.paginator import Paginator


from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect
from django.utils import timezone
from django.views.decorators.http import require_POST

from datetime import timedelta
from decimal import Decimal
import calendar
from datetime import datetime, timezone as dt_timezone

import ccxt
import pandas as pd
import numpy as np
from openai import OpenAI

from django.urls import reverse
from urllib.parse import urlencode

from nacl.signing import VerifyKey
from nacl.exceptions import BadSignatureError

from django.template import TemplateDoesNotExist
from django.template.loader import get_template

from .forms import ProfileForm
from .models import (
    TokenSnapshot,
    SponsoredProject,
    UserProfile,
    UserAttendance,
    SponsoredAdSubmission,
    # Campaign models
    Campaign,
    CampaignField,
    CampaignSubmission,
    CampaignSubmissionValue,
    CampaignSubmissionImage,
    CampaignPayoutRequest,
    CampaignPayoutLine,
    Announcement,
    MarketStudySubscription,
    MarketStudyPost,
    HourlyCryptoPrediction,
    Game, GameSession, Participation, Winner, InvalidBet, UserSolanaWallet
)

import market.engine as fomo_engine

# ============================================================
# Mobile / PC template helper
# ============================================================
def _is_mobile(request):
    ua = (request.META.get("HTTP_USER_AGENT") or "").lower()
    keywords = ("mobile", "android", "iphone", "ipad", "ipod", "webos",
                "blackberry", "opera mini", "opera mobi", "windows phone")
    return any(kw in ua for kw in keywords)


def render_m_pc(request, template_base, context=None, **kwargs):
    """
    Auto-select mobile or desktop template.

    Usage:
        render_m_pc(request, "market/profile", {...})
        → mobile:  "market/profile_mobile.html"
        → desktop: "market/profile.html"

    Falls back to desktop if the mobile template doesn't exist.
    """
    if _is_mobile(request):
        mobile_path = f"{template_base}_mobile.html"
        try:
            get_template(mobile_path)
            return render(request, mobile_path, context, **kwargs)
        except TemplateDoesNotExist:
            pass
    return render(request, f"{template_base}.html", context, **kwargs)

# ============================================================
# Home
# ============================================================
def home(request):
    majors_qs = (
        TokenSnapshot.objects
        .filter(category="major")
        .order_by("-market_cap", "-volume_24h")
    )

    potential_qs = (
        TokenSnapshot.objects
        .filter(category="potential")
        .order_by("-potential_score_hits", "-vol_mcap_ratio", "-volume_24h")
    )

    now = timezone.now()
    sponsored_qs = (
        SponsoredProject.objects
        .filter(is_active=True)
        .filter(models.Q(expires_at__isnull=True) | models.Q(expires_at__gt=now))
        .annotate(
            tier_rank=Case(
                When(tier="premium", then=Value(0)),
                When(tier="basic", then=Value(1)),
                default=Value(9),
                output_field=IntegerField(),
            ),
            mcap_sort=Coalesce("mcap_usd", Value(0.0)),
        )
        .order_by("tier_rank", "-mcap_sort", "order", "id")
    )

    sponsored_paginator = Paginator(sponsored_qs, 2)
    majors_paginator = Paginator(majors_qs, 5)
    potential_paginator = Paginator(potential_qs, 10)

    sp = request.GET.get("sp", 1)
    m = request.GET.get("m", 1)
    p = request.GET.get("p", 1)

    return render(request, "market/home.html", {
        "sponsored_page": sponsored_paginator.get_page(sp),
        "majors_page": majors_paginator.get_page(m),
        "potential_page": potential_paginator.get_page(p),
    })


def about(request):
    return render_m_pc(request, "market/about")


# ============================================================
# Phantom login
# ============================================================
ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def b58decode(s: str) -> bytes:
    n = 0
    for c in s:
        n *= 58
        if c not in ALPHABET:
            raise ValueError("Invalid base58")
        n += ALPHABET.index(c)
    h = n.to_bytes((n.bit_length() + 7) // 8, "big") or b"\x00"
    pad = 0
    for c in s:
        if c == "1":
            pad += 1
        else:
            break
    return b"\x00" * pad + h


def _get_client_ip(request) -> str | None:
    xff = request.META.get("HTTP_X_FORWARDED_FOR")
    if xff:
        return xff.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR")


def _require_login(request):
    if not request.session.get("wallet_address"):
        return redirect("home")
    return None


@require_GET
def phantom_nonce(request):
    nonce = secrets.token_urlsafe(16)
    request.session["phantom_nonce"] = nonce
    msg = f"LinkHash login\nNonce: {nonce}"
    return JsonResponse({"message": msg})


@csrf_exempt
@require_POST
def phantom_verify(request):
    """
    body: { publicKey: "<base58>", signature: "<base58>", message: "..." }
    """
    try:
        data = json.loads(request.body.decode("utf-8"))
        public_key = (data.get("publicKey") or "").strip()
        signature = (data.get("signature") or "").strip()
        message = data.get("message") or ""

        if not public_key or not signature or not message:
            return JsonResponse({"ok": False, "error": "Missing fields"}, status=400)

        nonce = request.session.get("phantom_nonce")
        if not nonce:
            return JsonResponse({"ok": False, "error": "Missing nonce"}, status=400)

        expected = f"LinkHash login\nNonce: {nonce}"
        if message != expected:
            return JsonResponse({"ok": False, "error": "Message mismatch"}, status=400)

        pk_bytes = b58decode(public_key)
        sig_bytes = b58decode(signature)
        vk = VerifyKey(pk_bytes)
        vk.verify(message.encode("utf-8"), sig_bytes)

        request.session["wallet_address"] = public_key
        request.session.pop("phantom_nonce", None)

        ip = _get_client_ip(request)
        now = timezone.now()

        user, created = UserProfile.objects.get_or_create(
            wallet_address=public_key,
            defaults={"last_login_at": now, "last_seen_ip": ip}
        )
        if not created:
            UserProfile.objects.filter(pk=user.pk).update(
                last_login_at=now,
                last_seen_ip=ip,
            )

        return JsonResponse({"ok": True})

    except BadSignatureError:
        return JsonResponse({"ok": False, "error": "Bad signature"}, status=401)
    except json.JSONDecodeError:
        return JsonResponse({"ok": False, "error": "Invalid JSON"}, status=400)
    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)


def logout_view(request):
    request.session.pop("wallet_address", None)
    request.session.pop("phantom_nonce", None)
    return redirect("home")


# ============================================================
# Profile
# ============================================================
def utc_today_date():
    return datetime.now(dt_timezone.utc).date()

@require_http_methods(["GET", "POST"])
def profile(request):
    gate = _require_login(request)
    if gate:
        return gate

    wallet = request.session.get("wallet_address")
    user, _ = UserProfile.objects.get_or_create(wallet_address=wallet)

    # ✅ UTC 기준 오늘 날짜
    today_utc = datetime.now(dt_timezone.utc).date()

    is_edit = request.GET.get("edit") == "1"

    # =========================
    # POST 처리
    # - 출석체크 (action=checkin)
    # - 프로필 저장 (기존)
    # =========================
    if request.method == "POST":
        action = request.POST.get("action", "")

        # ✅ 1) 출석체크
        if action == "checkin":
            with transaction.atomic():
                up = UserProfile.objects.select_for_update().get(pk=user.pk)
                att, created = UserAttendance.objects.get_or_create(
                    user=up,
                    date_utc=today_utc,
                )
                if created:
                    up.points = (up.points or 0) + 1
                    up.save(update_fields=["points"])
                    messages.success(request, "✅ Checked in! +1 point.")
                else:
                    messages.info(request, "You already checked in today (UTC).")

            return redirect("profile")

        # ✅ 2) 프로필 저장 (기존 로직 유지)
        form = ProfileForm(request.POST, request.FILES, instance=user)
        if form.is_valid():
            obj = form.save(commit=False)

            clear_requested = (request.POST.get("clear_avatar") == "1")
            if clear_requested:
                if getattr(obj, "avatar", None):
                    obj.avatar.delete(save=False)
                obj.avatar = None

            obj.save()
            return redirect("profile")
        else:
            is_edit = True
    else:
        form = ProfileForm(instance=user)

    # =========================
    # Creator campaigns (기존)
    # =========================
    my_campaigns_qs = Campaign.objects.filter(created_by_wallet=wallet).order_by("-created_at")
    my_campaigns_paginator = Paginator(my_campaigns_qs, 5)
    mc = request.GET.get("mc", 1)  # my campaigns page param
    my_campaigns = my_campaigns_paginator.get_page(mc)

    # =========================
    # Attendance calendar (이번 달, UTC 기준)
    # =========================
    year = today_utc.year
    month = today_utc.month
    first_weekday, days_in_month = calendar.monthrange(year, month)  # Monday=0

    checked_dates = set(
        UserAttendance.objects.filter(
            user=user,
            date_utc__year=year,
            date_utc__month=month,
        ).values_list("date_utc", flat=True)
    )

    weeks = []
    week = []

    # 앞쪽 빈칸
    for _ in range(first_weekday):
        week.append({"day": None, "status": "empty"})

    for d in range(1, days_in_month + 1):
        day_date = datetime(year, month, d, tzinfo=dt_timezone.utc).date()

        if day_date in checked_dates:
            status = "checked"
        else:
            if day_date <= today_utc:
                status = "missed"
            else:
                status = "future"

        week.append({"day": d, "status": status})

        if len(week) == 7:
            weeks.append(week)
            week = []

    # 끝쪽 빈칸
    if week:
        while len(week) < 7:
            week.append({"day": None, "status": "empty"})
        weeks.append(week)

    checked_in_today = today_utc in checked_dates

    # =========================
    # Market Study subscription
    # =========================
    now = timezone.now()
    ms_sub = (
        MarketStudySubscription.objects.filter(
            wallet_address=wallet,
            status=MarketStudySubscription.STATUS_VERIFIED,
            expires_at__gt=now,
        )
        .order_by("-expires_at")
        .first()
    )

    ms_subscription_active = bool(ms_sub)
    ms_expires_at = ms_sub.expires_at if ms_sub else None

    return render_m_pc(request, "market/profile", {
        "wallet_address": wallet,
        "user_profile": user,
        "form": form,
        "is_edit": is_edit,
        "my_campaigns": my_campaigns,

        # ✅ attendance
        "attendance_year": year,
        "attendance_month": month,
        "attendance_weeks": weeks,
        "checked_in_today": checked_in_today,

        # market study subscription
        "ms_subscription_active": ms_subscription_active,
        "ms_expires_at": ms_expires_at,
    })


def partnership(request):
    return render(request, "market/partnership.html")


def settings_page(request):
    gate = _require_login(request)
    if gate:
        return gate
    return render(request, "market/settings.html", {
        "wallet_address": request.session.get("wallet_address"),
    })


# ============================================================
# Solana helpers
# ============================================================
def _solana_rpc(method: str, params: list):
    rpc_url = getattr(settings, "SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    res = requests.post(
        rpc_url,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        timeout=12,
    )
    res.raise_for_status()
    data = res.json()
    if "error" in data:
        raise Exception(data["error"])
    return data["result"]


def verify_lhx_payment(
    signature: str,
    payer_wallet: str,
    recipient_wallet: str,
    required_amount_base: int,
):
    """
    signature tx에서:
    - 성공 tx인지 (meta.err 없음)
    - payer_wallet -> recipient_wallet 로 (LHX mint)
    - amount (base units) >= required_amount_base
    확인
    """
    tx = _solana_rpc(
        "getTransaction",
        [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
    )

    if not tx:
        return {"ok": False, "error": "Transaction not found yet. Try again shortly."}

    meta = tx.get("meta")
    if not meta or meta.get("err") is not None:
        return {"ok": False, "error": "Transaction failed."}

    pre = meta.get("preTokenBalances") or []
    post = meta.get("postTokenBalances") or []

    mint = getattr(settings, "LINKHASH_LHX_MINT", "9LrT8gAKJ5qUJA1wJoQRrVfapJGrNnU2ca5UYiJipump")

    def _sum_amount(items, owner):
        total = 0
        for it in items:
            if (it.get("mint") or "").strip() != mint:
                continue
            if (it.get("owner") or "").strip() != owner:
                continue
            ui = it.get("uiTokenAmount") or {}
            amount = ui.get("amount")
            if amount is None:
                continue
            total += int(amount)  # base units
        return total

    pre_payer = _sum_amount(pre, payer_wallet)
    post_payer = _sum_amount(post, payer_wallet)

    pre_rec = _sum_amount(pre, recipient_wallet)
    post_rec = _sum_amount(post, recipient_wallet)

    payer_delta = pre_payer - post_payer
    rec_delta = post_rec - pre_rec

    paid_base = min(payer_delta, rec_delta)

    if paid_base <= 0:
        return {"ok": False, "error": "No LHX transfer detected."}

    if paid_base < required_amount_base:
        decimals = int(getattr(settings, "LINKHASH_LHX_DECIMALS", 6))
        human_paid = paid_base / (10 ** decimals)
        human_required = required_amount_base / (10 ** decimals)
        return {"ok": False, "error": f"Insufficient amount: paid {human_paid} LHX, required {human_required} LHX"}

    return {
        "ok": True,
        "amount_raw": paid_base,
        "mint": mint,
        "recipient": recipient_wallet,
        "sender": payer_wallet,
    }


# ============================================================
# Sponsored Ads submission (LHX payment -> auto create SponsoredProject)
# ============================================================

def _normalize_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return ""
    # allow mailto / tg deep links if you ever use them; otherwise keep it simple
    if u.startswith(("http://", "https://")):
        return u
    return "https://" + u


@require_POST
def submit_ad(request):
    gate = _require_login(request)
    if gate:
        return gate

    wallet = request.session.get("wallet_address")

    if not wallet:
        return JsonResponse({"ok": False, "error": "Not logged in."}, status=401)

    # =========================
    # Basic inputs
    # =========================
    tier = (request.POST.get("tier") or "").strip().lower()
    tx_signature = (request.POST.get("tx_signature") or "").strip()

    project_name = (request.POST.get("project_name") or "").strip()
    symbol = (request.POST.get("symbol") or "").strip()
    token_address = (request.POST.get("token_address") or "").strip()
    website = _normalize_url(request.POST.get("website"))
    dex_url = _normalize_url(request.POST.get("dex_url"))  # ✅ 기존 유지 (DexScreener/auto sync 용)
    x_url = _normalize_url(request.POST.get("x_url"))
    tg_url = _normalize_url(request.POST.get("tg_url"))
    description = (request.POST.get("description") or "").strip()

    banner_image = request.FILES.get("banner_image")

    # =========================
    # ✅ NEW: optional multi links
    # - HTML에서 name="dex_links[]" / "cex_links[]" 로 보낸다고 가정
    # =========================
    def _normalize_many(raw_list, max_n):
        items = []
        for v in (raw_list or []):
            u = _normalize_url(v)
            if u:
                items.append(u)
        # dedupe (order-preserving)
        seen = set()
        uniq = []
        for u in items:
            if u in seen:
                continue
            seen.add(u)
            uniq.append(u)
        return uniq[:max_n]

    dex_links = _normalize_many(request.POST.getlist("dex_links[]"), 3)  # ✅ max 3
    cex_links = _normalize_many(request.POST.getlist("cex_links[]"), 10)  # ✅ max 10

    # ✅ 호환: 기존 dex_url input이 있고 dex_links가 비어있으면 dex_links[0]로 넣어주기
    # (기존 폼/유저 습관 유지)
    if not dex_links and dex_url:
        dex_links = [dex_url]
    # ✅ 반대로, dex_url이 비어있고 dex_links[0]가 있으면 dex_url을 채워서 기존 로직 유지
    if not dex_url and dex_links:
        dex_url = dex_links[0]

    # =========================
    # NEW: chain / launchpad
    # =========================
    chain = (request.POST.get("chain") or SponsoredAdSubmission.CHAIN_SOL).strip().lower()
    launchpad = (request.POST.get("launchpad") or SponsoredAdSubmission.LAUNCHPAD_NONE).strip().lower()
    graduated_raw = (request.POST.get("graduated") or "").strip().lower()
    graduated = graduated_raw in ("1", "true", "yes", "on")

    # =========================
    # Validation
    # =========================
    if tier not in (SponsoredAdSubmission.TIER_BASIC, SponsoredAdSubmission.TIER_PREMIUM):
        return JsonResponse({"ok": False, "error": "Invalid tier"}, status=400)

    if chain not in (SponsoredAdSubmission.CHAIN_SOL, SponsoredAdSubmission.CHAIN_BNB, SponsoredAdSubmission.CHAIN_OTHERS):
        return JsonResponse({"ok": False, "error": "Invalid chain"}, status=400)

    if launchpad not in (
        SponsoredAdSubmission.LAUNCHPAD_NONE,
        SponsoredAdSubmission.LAUNCHPAD_PUMPFUN,
        SponsoredAdSubmission.LAUNCHPAD_FOURMEME,
    ):
        return JsonResponse({"ok": False, "error": "Invalid launchpad"}, status=400)

    # ✅ chain ↔ launchpad consistency rules
    if launchpad == SponsoredAdSubmission.LAUNCHPAD_PUMPFUN and chain != SponsoredAdSubmission.CHAIN_SOL:
        return JsonResponse({"ok": False, "error": "Pump.fun launchpad requires Solana chain."}, status=400)

    if launchpad == SponsoredAdSubmission.LAUNCHPAD_FOURMEME and chain != SponsoredAdSubmission.CHAIN_BNB:
        return JsonResponse({"ok": False, "error": "four.meme launchpad requires BNB chain."}, status=400)

    if graduated and launchpad == SponsoredAdSubmission.LAUNCHPAD_NONE:
        return JsonResponse({"ok": False, "error": "Graduated=true requires a launchpad selection."}, status=400)


    if not tx_signature or not project_name:
        return JsonResponse({"ok": False, "error": "Missing required fields"}, status=400)

    # Optional: basic size/type checks for banner
    if banner_image:
        max_bytes = int(getattr(settings, "LINKHASH_BANNER_MAX_BYTES", 5 * 1024 * 1024))  # 5MB default
        if banner_image.size > max_bytes:
            return JsonResponse({"ok": False, "error": "Banner image too large."}, status=400)
        content_type = (getattr(banner_image, "content_type", "") or "").lower()
        if content_type and content_type not in ("image/png", "image/jpeg", "image/jpg", "image/webp"):
            return JsonResponse({"ok": False, "error": "Invalid banner image type."}, status=400)

    DEV_WALLET = getattr(settings, "LINKHASH_DEV_WALLET", "")
    if not DEV_WALLET:
        return JsonResponse(
            {"ok": False, "error": "Server misconfigured: missing LINKHASH_DEV_WALLET"},
            status=500,
        )

    # =========================
    # LHX payment config
    # =========================
    LHX_DECIMALS = int(getattr(settings, "LINKHASH_LHX_DECIMALS", 6))
    BASIC_LHX = int(getattr(settings, "BASIC_AD_LHX", 100_000))
    PREMIUM_LHX = int(getattr(settings, "PREMIUM_AD_LHX", 250_000))
    BASIC_DAYS = int(getattr(settings, "BASIC_AD_DAYS", 14))
    PREMIUM_DAYS = int(getattr(settings, "PREMIUM_AD_DAYS", 60))

    required_lhx = BASIC_LHX if tier == SponsoredAdSubmission.TIER_BASIC else PREMIUM_LHX
    required_amount_base = required_lhx * (10 ** LHX_DECIMALS)

    if SponsoredAdSubmission.objects.filter(tx_signature=tx_signature).exists():
        return JsonResponse(
            {"ok": False, "error": "This transaction was already used."},
            status=400,
        )

    # =========================
    # On-chain verification
    # =========================
    try:
        verified = verify_lhx_payment(
            signature=tx_signature,
            payer_wallet=wallet,
            recipient_wallet=DEV_WALLET,
            required_amount_base=required_amount_base,
        )
        if not verified.get("ok"):
            return JsonResponse(
                {"ok": False, "error": verified.get("error", "Payment not verified")},
                status=400,
            )

        paid_base = int(verified.get("amount_raw") or 0)
        amount_lhx_int = paid_base // (10 ** LHX_DECIMALS)

        if amount_lhx_int < required_lhx:
            return JsonResponse(
                {"ok": False, "error": "Insufficient amount after parsing."},
                status=400,
            )

    except Exception as e:
        return JsonResponse(
            {"ok": False, "error": f"Verification error: {str(e)}"},
            status=400,
        )

    # =========================
    # Dates / ordering / points
    # =========================
    now = timezone.now()
    days = BASIC_DAYS if tier == SponsoredAdSubmission.TIER_BASIC else PREMIUM_DAYS
    expires_at = now + timezone.timedelta(days=days)
    tier_priority = 0 if tier == SponsoredAdSubmission.TIER_PREMIUM else 1

    # points
    points_delta = 15 if tier == SponsoredAdSubmission.TIER_BASIC else 30

    # =========================
    # DB transaction
    # =========================
    try:
        with transaction.atomic():
            sub = SponsoredAdSubmission.objects.create(
                status=SponsoredAdSubmission.STATUS_VERIFIED,
                tier=tier,
                submitter_wallet=wallet,
                recipient_wallet=DEV_WALLET,
                tx_signature=tx_signature,
                amount_lhx=amount_lhx_int,
                verified_at=now,
                expires_at=expires_at,

                project_name=project_name,
                symbol=symbol,
                token_address=token_address,
                website=website,
                dex_url=dex_url,
                x_url=x_url,
                tg_url=tg_url,
                description=description,
                # ✅ NEW multi links
                cex_links=cex_links,
                dex_links=dex_links,
                # NEW
                chain=chain,
                launchpad=launchpad,
                graduated=graduated,

                banner_image=banner_image,
            )

            sp = SponsoredProject.objects.create(
                is_active=True,
                order=tier_priority,
                name=project_name,
                symbol=symbol,
                token_address=token_address,
                website=website,
                dex_url=dex_url,
                x_url=x_url,
                tg_url=tg_url,
                description=description,
                cex_links=cex_links,
                dex_links=dex_links,

                # NEW
                chain=chain,
                launchpad=launchpad,
                graduated=graduated,

                expires_at=expires_at,
                tier=tier,
                source_wallet=wallet,
            )

            if sub.banner_image:
                sp.image = sub.banner_image
                sp.save(update_fields=["image"])

            # points
            UserProfile.objects.get_or_create(wallet_address=wallet)
            UserProfile.objects.filter(wallet_address=wallet).update(
                points=Coalesce(F("points"), 0) + points_delta
            )

    except IntegrityError:
        return JsonResponse({"ok": False, "error": "Duplicate submission."}, status=400)

    return JsonResponse({"ok": True, "points_awarded": points_delta})


# ============================================================
# Campaigns (list / detail / submit)
# ============================================================
def _sync_ended_campaigns():
    now = timezone.now()
    Campaign.objects.filter(
        status=Campaign.STATUS_ACTIVE,
        ends_at__isnull=False,
        ends_at__lt=now,
    ).update(status=Campaign.STATUS_ENDED)


def _require_creator(request, campaign: Campaign):
    wallet = request.session.get("wallet_address")
    if not wallet:
        return None, redirect("home")  # 로그인 페이지가 따로면 거기로
    if (campaign.created_by_wallet or "").strip() != wallet:
        return wallet, HttpResponseForbidden("Not allowed")
    return wallet, None

@require_GET
def creator_campaign_list(request):
    gate = _require_login(request)
    if gate:
        return gate

    wallet = request.session.get("wallet_address")

    qs = Campaign.objects.filter(
        created_by_wallet=wallet
    ).order_by("-created_at")

    # ✅ pagination (5 per page)
    page_num = request.GET.get("mc", 1)
    paginator = Paginator(qs, 5)
    page_obj = paginator.get_page(page_num)

    return render(request, "market/creator/campaign_list.html", {
        "campaigns": page_obj,   # ⚠️ Page object
        "wallet_address": wallet,
    })


def campaign_list(request):
    _sync_ended_campaigns()
    now = timezone.now()

    active_qs = (
        Campaign.objects
        .filter(status=Campaign.STATUS_ACTIVE)
        .filter(models.Q(starts_at__isnull=True) | models.Q(starts_at__lte=now))
        .filter(models.Q(ends_at__isnull=True) | models.Q(ends_at__gte=now))
        .order_by("-created_at")
    )

    past_qs = (
        Campaign.objects
        .filter(status=Campaign.STATUS_ENDED)
        .order_by("-ends_at", "-created_at")
    )

    # ✅ 페이지 파라미터: active=cp, past=pp
    cp = request.GET.get("cp", 1)
    pp = request.GET.get("pp", 1)

    # ✅ 3개씩
    active_paginator = Paginator(active_qs, 3)
    past_paginator = Paginator(past_qs, 3)

    return render(request, "market/campaigns/list.html", {
        # ✅ 템플릿 변수명 그대로 유지 (중요)
        "campaigns": active_paginator.get_page(cp),
        "past_campaigns": past_paginator.get_page(pp),
    })

def campaign_detail(request, campaign_id: int):
    _sync_ended_campaigns()
    campaign = get_object_or_404(Campaign, pk=campaign_id)  # ✅ ended도 접근 가능

    fields = campaign.fields.all().order_by("order", "id")

    fee_wallet = getattr(
        settings,
        "CAMPAIGN_FEE_WALLET",
        getattr(settings, "LINKHASH_DEV_WALLET", "")
    )

    submissions = campaign.submissions.all().order_by("-created_at")[:30]

    is_past = (campaign.status == Campaign.STATUS_ENDED)

    return render(request, "market/campaigns/detail.html", {
        "campaign": campaign,
        "fields": fields,
        "fee_lhx": campaign.submission_fee_lhx,
        "fee_wallet": fee_wallet,
        "submissions": submissions,
        "is_past": is_past,  # ✅ 템플릿에서 submit 비활성화
    })


from django.http import HttpResponse
from django.template.loader import render_to_string

@require_POST
def campaign_submit(request, campaign_id: int):
    gate = _require_login(request)
    if gate:
        return gate

    campaign = get_object_or_404(Campaign, pk=campaign_id, status=Campaign.STATUS_ACTIVE)
    wallet = request.session.get("wallet_address")

    fee_tx = (request.POST.get("fee_tx_signature") or "").strip()
    if not fee_tx:
        return _hx_or_json_error(request, "Missing fee tx signature", status=400)

    if CampaignSubmission.objects.filter(fee_tx_signature=fee_tx).exists():
        return _hx_or_json_error(request, "This fee transaction was already used.", status=400)

    LHX_DECIMALS = int(getattr(settings, "LINKHASH_LHX_DECIMALS", 6))
    required_lhx = int(getattr(settings, "CAMPAIGN_SUBMIT_LHX_FEE", campaign.submission_fee_lhx))
    required_amount_base = required_lhx * (10 ** LHX_DECIMALS)

    # ✅ 이 줄은 중복이라 고쳐야 함 (현재 getattr 두번 동일키)
    FEE_WALLET = getattr(settings, "CAMPAIGN_FEE_WALLET", getattr(settings, "LINKHASH_DEV_WALLET", ""))
    if not FEE_WALLET:
        return _hx_or_json_error(request, "Server misconfigured: missing CAMPAIGN_FEE_WALLET", status=500)

    verified = verify_lhx_payment(
        signature=fee_tx,
        payer_wallet=wallet,
        recipient_wallet=FEE_WALLET,
        required_amount_base=required_amount_base,
    )
    if not verified.get("ok"):
        return _hx_or_json_error(request, verified.get("error", "Fee not verified"), status=400)

    paid_base = int(verified.get("amount_raw") or 0)
    paid_lhx_int = paid_base // (10 ** LHX_DECIMALS)

    fields = list(campaign.fields.all().order_by("order", "id"))
    images = request.FILES.getlist("images")

    now = timezone.now()
    try:
        with transaction.atomic():
            sub = CampaignSubmission.objects.create(
                campaign=campaign,
                submitter_wallet=wallet,
                fee_paid_lhx=paid_lhx_int,
                fee_tx_signature=fee_tx,
                fee_verified_at=now,
                status=CampaignSubmission.STATUS_PENDING,
            )

            for f in fields:
                v = (request.POST.get(f"field__{f.key}") or "").strip()
                if f.required and not v:
                    raise ValueError(f"Missing required field: {f.key}")
                CampaignSubmissionValue.objects.create(submission=sub, field=f, value_text=v)

            for img in images:
                CampaignSubmissionImage.objects.create(submission=sub, image=img)

    except ValueError as e:
        return _hx_or_json_error(request, str(e), status=400)

    # ✅ HTMX 요청이면: 성공 메시지 HTML 반환 (페이지 이동 없음)
    if request.headers.get("HX-Request") == "true":
        html = render_to_string("market/campaigns/_submit_success.html", {
            "campaign": campaign,
            "submitter_wallet": wallet,
        }, request=request)
        return HttpResponse(html)

    return JsonResponse({"ok": True})



from django.views.decorators.http import require_GET

@require_GET
def campaign_submissions_partial(request, campaign_id: int):
    campaign = get_object_or_404(Campaign, pk=campaign_id, status=Campaign.STATUS_ACTIVE)
    submissions = campaign.submissions.all().order_by("-created_at")[:30]
    html = render_to_string("market/campaigns/_submissions_list.html", {
        "campaign": campaign,
        "submissions": submissions,
    }, request=request)
    return HttpResponse(html)


def _hx_or_json_error(request, msg: str, status: int = 400, code: str = ""):
    is_hx = request.headers.get("HX-Request") == "true"
    if is_hx:
        html = render_to_string("market/campaigns/_submit_error.html",
                                {"error": msg, "code": code},
                                request=request)
        # ✅ HTMX에서는 200으로 보내서 무조건 swap 되게
        return HttpResponse(html, status=200)
    return JsonResponse({"ok": False, "error": msg, "code": code}, status=status)


# ============================================================
# Campaign create (MVP placeholder to fix ImportError)
# - GET: render create page
# - POST: create campaign (on-chain SOL verify is TODO)
# ============================================================
# ============================================================
# Solana: verify SOL transfer (for campaign pool funding)
# ============================================================
# ============================================================
# Campaign create (SOL funding verify + create)
# ============================================================

def _to_decimal_s(s: str):
    """
    DecimalField(max_digits=18, decimal_places=9) 안전 파서
    """
    from decimal import Decimal, InvalidOperation
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def _parse_dt_local(s: str):
    """
    <input type="datetime-local"> 값 파싱 (예: 2026-02-02T12:34)
    timezone-aware로 저장하고 싶으면 make_aware 권장.
    여기선 단순히 현재 timezone으로 aware 처리.
    """
    if not s:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        dt = timezone.datetime.fromisoformat(s)
        if timezone.is_naive(dt):
            dt = timezone.make_aware(dt, timezone.get_current_timezone())
        return dt
    except Exception:
        return None


def verify_sol_payment(
    signature: str,
    payer_wallet: str,
    recipient_wallet: str,
    required_lamports: int,
):
    """
    Solana tx(signature)가 다음을 만족하는지 확인:
    - 성공 tx(meta.err 없음)
    - payer_wallet -> recipient_wallet 로 SOL 이동이 존재
    - 이동 lamports >= required_lamports
    """
    try:
        tx = _solana_rpc(
            "getTransaction",
            [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
        )
    except Exception as e:
        return {"ok": False, "error": f"RPC error: {str(e)}"}

    if not tx:
        return {"ok": False, "error": "Transaction not found yet. Try again shortly."}

    meta = tx.get("meta")
    if not meta or meta.get("err") is not None:
        return {"ok": False, "error": "Transaction failed."}

    message = (tx.get("transaction") or {}).get("message") or {}
    account_keys = message.get("accountKeys") or []
    # accountKeys가 dict 형태로 올 수도 있음(jsonParsed)
    keys = []
    for k in account_keys:
        if isinstance(k, dict):
            keys.append(k.get("pubkey"))
        else:
            keys.append(k)

    # payer가 accounts[0]이 아닌 경우도 있어서, instruction / balances로 추적
    pre_bal = meta.get("preBalances") or []
    post_bal = meta.get("postBalances") or []

    # balances diff로 recipient 증가량 / payer 감소량 계산 (단, fee 영향으로 payer 감소량이 더 큼)
    # recipient_wallet index 찾기
    def idx_of(wallet):
        try:
            return keys.index(wallet)
        except ValueError:
            return -1

    payer_i = idx_of(payer_wallet)
    rec_i = idx_of(recipient_wallet)

    if payer_i == -1 or rec_i == -1:
        # fallback: instructions에서 system transfer 찾기
        instrs = message.get("instructions") or []
        transferred = 0
        for ix in instrs:
            parsed = ix.get("parsed") if isinstance(ix, dict) else None
            if not parsed:
                continue
            if parsed.get("type") != "transfer":
                continue
            info = parsed.get("info") or {}
            src = info.get("source")
            dst = info.get("destination")
            lamports = info.get("lamports")
            if src == payer_wallet and dst == recipient_wallet and lamports:
                transferred += int(lamports)

        if transferred < required_lamports:
            sol_paid = transferred / 1_000_000_000
            sol_req = required_lamports / 1_000_000_000
            return {"ok": False, "error": f"Insufficient SOL: paid {sol_paid} SOL, required {sol_req} SOL"}
        return {"ok": True, "lamports": transferred}

    if rec_i >= len(pre_bal) or rec_i >= len(post_bal):
        return {"ok": False, "error": "Unable to parse balances (recipient index out of range)."}

    rec_delta = int(post_bal[rec_i]) - int(pre_bal[rec_i])

    if rec_delta < required_lamports:
        sol_paid = rec_delta / 1_000_000_000
        sol_req = required_lamports / 1_000_000_000
        return {"ok": False, "error": f"Insufficient SOL: paid {sol_paid} SOL, required {sol_req} SOL"}

    return {"ok": True, "lamports": rec_delta}


@require_http_methods(["GET", "POST"])
def campaign_create(request):
    gate = _require_login(request)
    if gate:
        return gate

    platform_fee_pct = getattr(settings, "CAMPAIGN_PLATFORM_FEE_PCT", 5)
    submit_fee_lhx = int(getattr(settings, "CAMPAIGN_SUBMIT_LHX_FEE", 100))

    fee_wallet = getattr(
        settings,
        "CAMPAIGN_FEE_WALLET",
        getattr(settings, "LINKHASH_DEV_WALLET", "")
    )

    if request.method == "GET":
        return render(request, "market/campaigns/create.html", {
            "platform_fee_pct": platform_fee_pct,
            "submit_fee_lhx": submit_fee_lhx,
            "fee_wallet": fee_wallet,
        })

    # -----------------------
    # POST: Verify SOL + Create
    # -----------------------
    wallet = request.session.get("wallet_address")

    title = (request.POST.get("title") or "").strip()
    campaign_type = (request.POST.get("campaign_type") or "").strip()
    description = (request.POST.get("description") or "").strip()

    # ✅ NEW: duration 옵션 (7/14/30)
    duration_raw = (request.POST.get("duration_days") or "7").strip()
    try:
        duration_days = int(duration_raw)
    except ValueError:
        duration_days = 7
    if duration_days not in (7, 14, 30):
        duration_days = 7

    # ✅ NEW: cover image
    cover_image = request.FILES.get("cover_image")

    pool_sol = _to_decimal_s(request.POST.get("pool_sol"))
    reward_per_user_sol = _to_decimal_s(request.POST.get("reward_per_user_sol"))

    field1_label = (request.POST.get("field1_label") or "").strip()
    field1_key = (request.POST.get("field1_key") or "").strip()
    field2_label = (request.POST.get("field2_label") or "").strip()
    field2_key = (request.POST.get("field2_key") or "").strip()

    funding_tx_signature = (request.POST.get("funding_tx_signature") or "").strip()

    # (기존 starts_at/ends_at 입력은 더 이상 받지 않음. duration 기반으로 자동 설정)
    starts_at_dt = None
    ends_at_dt = None

    # validation
    if not fee_wallet:
        return JsonResponse(
            {"ok": False, "error": "Server misconfigured: missing CAMPAIGN_FEE_WALLET (or LINKHASH_DEV_WALLET)"},
            status=500)
    
    if not title or campaign_type not in (Campaign.TYPE_DEX_PROMO, Campaign.TYPE_GENERAL):
        return JsonResponse({"ok": False, "error": "Missing/invalid title or type"}, status=400)

    # ✅ NEW: pool 최소 0.05 SOL
    MIN_POOL = Decimal("0.05")
    if pool_sol is None or pool_sol < MIN_POOL:
        return JsonResponse({"ok": False, "error": "Reward pool must be at least 0.05 SOL."}, status=400)

    if not field1_label or not field1_key:
        return JsonResponse({"ok": False, "error": "Field #1 is required"}, status=400)

    import re
    def normalize_key(k):
        k = (k or "").strip().lower()
        k = re.sub(r"[^a-z0-9_\-]", "_", k)
        k = re.sub(r"_+", "_", k).strip("_")
        return k[:40]

    field1_key_n = normalize_key(field1_key)
    field2_key_n = normalize_key(field2_key) if field2_key else ""

    if not field1_key_n:
        return JsonResponse({"ok": False, "error": "Invalid Field #1 key"}, status=400)
    if field2_key and not field2_key_n:
        return JsonResponse({"ok": False, "error": "Invalid Field #2 key"}, status=400)
    if field2_key_n and field2_key_n == field1_key_n:
        return JsonResponse({"ok": False, "error": "Field #2 key must be different from Field #1 key"}, status=400)

    if not funding_tx_signature:
        return JsonResponse({"ok": False, "error": "Missing funding tx signature"}, status=400)

    if Campaign.objects.filter(create_tx_signature=funding_tx_signature).exists():
        return JsonResponse({"ok": False, "error": "This funding transaction was already used."}, status=400)

    # required SOL = pool * (1 + fee%)
    fee_pct_dec = Decimal(str(platform_fee_pct)) / Decimal("100")
    required_total_sol = pool_sol * (Decimal("1") + fee_pct_dec)
    required_lamports = int((required_total_sol * Decimal("1000000000")).to_integral_value())

    verified = verify_sol_payment(
        signature=funding_tx_signature,
        payer_wallet=wallet,
        recipient_wallet=fee_wallet,
        required_lamports=required_lamports,
    )
    if not verified.get("ok"):
        return JsonResponse({"ok": False, "error": verified.get("error", "Funding not verified")}, status=400)

    now = timezone.now()

    # ✅ NEW: duration 기반으로 starts_at/ends_at 자동
    starts_at_dt = now
    ends_at_dt = now + timedelta(days=duration_days)

    try:
        with transaction.atomic():
            c = Campaign.objects.create(
                title=title,
                description=description,
                campaign_type=campaign_type,
                status=Campaign.STATUS_ACTIVE,

                created_by_wallet=wallet,
                is_admin_created=False,

                starts_at=starts_at_dt,
                ends_at=ends_at_dt,

                # ✅ duration_days 필드가 모델에 있어야 함 (마이그레이션 필요)
                duration_days=duration_days,

                # ✅ cover image 저장
                cover_image=cover_image,

                submission_fee_lhx=submit_fee_lhx,

                pool_total_sol=pool_sol,
                reward_per_user_sol=reward_per_user_sol,
                distribute_mode="per_user" if reward_per_user_sol else "split_n",
                platform_fee_pct=Decimal(str(platform_fee_pct)),

                create_tx_signature=funding_tx_signature,
                create_verified_at=now,
            )

            CampaignField.objects.create(
                campaign=c,
                key=field1_key_n,
                label=field1_label,
                required=True,
                order=0,
                field_type=CampaignField.FIELD_TEXT,
            )

            if field2_label and field2_key_n:
                CampaignField.objects.create(
                    campaign=c,
                    key=field2_key_n,
                    label=field2_label,
                    required=False,
                    order=1,
                    field_type=CampaignField.FIELD_TEXT,
                )

            UserProfile.objects.get_or_create(wallet_address=wallet)
            UserProfile.objects.filter(wallet_address=wallet).update(points=F("points") + 25)

    except Exception as e:
        return JsonResponse({"ok": False, "error": str(e)}, status=400)

    return redirect("campaign_detail", campaign_id=c.id)


@require_POST
def campaign_create_verify(request):
    """
    (옵션) 프론트에서 '검증만' 하려고 쓰는 엔드포인트.
    현재 UI는 Verify & Create를 campaign_create로 보내므로
    이 엔드포인트는 유지하되, 간단 검증만 제공.
    """
    gate = _require_login(request)
    if gate:
        return gate

    wallet = request.session.get("wallet_address")
    tx = (request.POST.get("funding_tx_signature") or request.POST.get("pool_tx_signature") or "").strip()
    pool_sol = _to_decimal_s(request.POST.get("pool_sol"))

    platform_fee_pct = getattr(settings, "CAMPAIGN_PLATFORM_FEE_PCT", 5)
    fee_wallet = getattr(
        settings,
        "CAMPAIGN_FEE_WALLET",
        getattr(settings, "LINKHASH_DEV_WALLET", "")
    )

    if not tx:
        return JsonResponse({"ok": False, "error": "Missing tx signature"}, status=400)
    if pool_sol is None or pool_sol <= 0:
        return JsonResponse({"ok": False, "error": "Invalid pool_sol"}, status=400)
    if not fee_wallet:
        return JsonResponse({"ok": False, "error": "Server misconfigured: missing CAMPAIGN_FEE_WALLET"}, status=500)

    from decimal import Decimal
    fee_pct_dec = Decimal(str(platform_fee_pct)) / Decimal("100")
    required_total_sol = pool_sol * (Decimal("1") + fee_pct_dec)
    required_lamports = int((required_total_sol * Decimal("1000000000")).to_integral_value())

    verified = verify_sol_payment(
        signature=tx,
        payer_wallet=wallet,
        recipient_wallet=fee_wallet,
        required_lamports=required_lamports,
    )
    if not verified.get("ok"):
        return JsonResponse({"ok": False, "error": verified.get("error", "Funding not verified")}, status=400)

    return JsonResponse({"ok": True, "verified": True})


@require_GET
def creator_campaign_manage(request, campaign_id: int):
    gate = _require_login(request)
    if gate:
        return gate

    campaign = get_object_or_404(Campaign, pk=campaign_id)
    wallet, forbid = _require_creator(request, campaign)
    if forbid:
        return forbid

    fields = list(campaign.fields.all().order_by("order", "id"))

    # ✅ slice 제거: 전체 submissions를 pagination으로
    submissions_qs = (
        campaign.submissions
        .all()
        .prefetch_related("values__field", "images")
        .order_by("-created_at")
    )

    per_page = 10  # 원하면 25/50으로 바꿔도 됨
    paginator = Paginator(submissions_qs, per_page)
    sp = request.GET.get("sp", "1")
    submissions = paginator.get_page(sp)

    last_req = None
    if hasattr(campaign, "payout_requests"):
        last_req = campaign.payout_requests.order_by("-created_at").first()

    return render(request, "market/creator/campaign_manage.html", {
        "campaign": campaign,
        "fields": fields,
        "submissions": submissions,  # Page 객체
        "last_req": last_req,
        "wallet_address": wallet,
    })


@require_POST
def creator_mark_submission(request, campaign_id: int):
    gate = _require_login(request)
    if gate:
        return gate

    campaign = get_object_or_404(Campaign, pk=campaign_id)
    wallet, forbid = _require_creator(request, campaign)
    if forbid:
        return forbid

    sub_id = (request.POST.get("submission_id") or "").strip()
    action = (request.POST.get("action") or "").strip().lower()
    note = (request.POST.get("note") or "").strip()

    if not sub_id:
        messages.error(request, "Missing submission_id")
        return redirect("creator_campaign_manage", campaign_id=campaign.id)

    sub = get_object_or_404(CampaignSubmission, pk=sub_id, campaign=campaign)

    # ✅ 1) NOTE만 저장
    if action == "note":
        sub.reviewer_note = note
        if hasattr(sub, "reviewed_by_wallet"):
            sub.reviewed_by_wallet = wallet
        if hasattr(sub, "reviewed_at"):
            sub.reviewed_at = timezone.now()
        sub.save(update_fields=["reviewer_note", "reviewed_by_wallet", "reviewed_at"] if hasattr(sub,
                                                                                                 "reviewed_by_wallet") else [
            "reviewer_note", "reviewed_at"])
        
        messages.success(request, "Saved note.")
        return redirect("creator_campaign_manage", campaign_id=campaign.id)

    # ✅ 2) STATUS 변경
    if action not in ("approve", "reject", "pending"):
        messages.error(request, "Invalid action")
        return redirect("creator_campaign_manage", campaign_id=campaign.id)

    if action == "approve":
        sub.status = CampaignSubmission.STATUS_APPROVED
    elif action == "reject":
        sub.status = CampaignSubmission.STATUS_REJECTED
    else:
        sub.status = CampaignSubmission.STATUS_PENDING

    # ✅ NOTE는 status 변경 시 덮어쓰지 않음 (중요)
    if hasattr(sub, "reviewed_by_wallet"):
        sub.reviewed_by_wallet = wallet
    if hasattr(sub, "reviewed_at"):
        sub.reviewed_at = timezone.now()

    # status만 저장
    fields = ["status"]
    if hasattr(sub, "reviewed_by_wallet"):
        fields.append("reviewed_by_wallet")
    if hasattr(sub, "reviewed_at"):
        fields.append("reviewed_at")

    sub.save(update_fields=fields)

    messages.success(request, "Updated submission status.")
    return redirect("creator_campaign_manage", campaign_id=campaign.id)


from decimal import Decimal

@require_POST
def creator_request_payout(request, campaign_id: int):
    gate = _require_login(request)
    if gate:
        return gate

    campaign = get_object_or_404(Campaign, pk=campaign_id)
    wallet, forbid = _require_creator(request, campaign)
    if forbid:
        return forbid

    # per_user만 MVP 지원
    if not campaign.reward_per_user_sol or campaign.reward_per_user_sol <= 0:
        messages.error(request, "reward_per_user_sol is not set for this campaign.")
        return redirect("creator_campaign_manage", campaign_id=campaign.id)

    per_user = campaign.reward_per_user_sol
    now = timezone.now()

    with transaction.atomic():
        # ✅ 최신 요청 잠그고 가져오기
        last_req = (
            CampaignPayoutRequest.objects
            .select_for_update()
            .filter(campaign=campaign)
            .order_by("-created_at")
            .first()
        )

        # ✅ requested 상태면: sync 모드(라인/예약 초기화 후 재계산)
        if last_req and last_req.status == CampaignPayoutRequest.STATUS_REQUESTED:
            req = last_req

            # 이 요청에 묶여있던 reserved submissions 풀기
            CampaignSubmission.objects.filter(
                payout_request=req,
                payout_status=CampaignSubmission.PAYOUT_RESERVED,
            ).update(
                payout_status=CampaignSubmission.PAYOUT_NONE,
                payout_request=None,
                payout_reserved_at=None,
            )

            # 기존 payout lines 삭제
            CampaignPayoutLine.objects.filter(request=req).delete()

            # 요청 메타 업데이트
            req.requested_by_wallet = wallet
            req.created_at = now
            req.memo = req.memo or ""
            req.total_recipients = 0
            req.total_amount_sol = Decimal("0")
            req.save(
                update_fields=["requested_by_wallet", "created_at", "memo", "total_recipients", "total_amount_sol"])
            
        else:
            # processing/paid/rejected면 새 요청 생성
            req = CampaignPayoutRequest.objects.create(
                campaign=campaign,
                requested_by_wallet=wallet,
                status=CampaignPayoutRequest.STATUS_REQUESTED,
                memo="",
                total_recipients=0,
                total_amount_sol=Decimal("0"),
                created_at=now,
            )

        # ✅ 현재 approved + payout_status none 만 대상
        eligible_rows = list(
            CampaignSubmission.objects
            .select_for_update()
            .filter(
                campaign=campaign,
                status=CampaignSubmission.STATUS_APPROVED,
                payout_status=CampaignSubmission.PAYOUT_NONE,
            )
            .values("id", "submitter_wallet")
            .order_by("created_at")
        )

        if not eligible_rows:
            messages.info(request, "No newly approved submissions to pay (already reserved/paid or none approved).")
            return redirect("creator_campaign_manage", campaign_id=campaign.id)

        eligible_ids = [r["id"] for r in eligible_rows]

        # ✅ 핵심: 이미 payout line이 존재하는 submission은 제외 (UNIQUE 충돌 방지)
        existing_line_ids = set(
            CampaignPayoutLine.objects
            .filter(submission_id__in=eligible_ids)
            .values_list("submission_id", flat=True)
        )

        if existing_line_ids:
            eligible_rows = [r for r in eligible_rows if r["id"] not in existing_line_ids]
            eligible_ids = [r["id"] for r in eligible_rows]

        if not eligible_rows:
            messages.info(
                request,
                "All approved submissions are already included in previous payout lines (nothing new to add)."
            )
            return redirect("creator_campaign_manage", campaign_id=campaign.id)

        # 1) submissions 예약 처리
        CampaignSubmission.objects.filter(id__in=eligible_ids).update(
            payout_status=CampaignSubmission.PAYOUT_RESERVED,
            payout_request=req,
            payout_reserved_at=now,
        )

        # 2) payout lines 생성
        lines = [
            CampaignPayoutLine(
                request=req,
                recipient_wallet=r["submitter_wallet"],
                amount_sol=per_user,
                submission_id=r["id"],
            )
            for r in eligible_rows
        ]
        CampaignPayoutLine.objects.bulk_create(lines)

        # 3) 합계 업데이트
        total_recipients = len(lines)
        total_amount = per_user * Decimal(total_recipients)

        req.total_recipients = total_recipients
        req.total_amount_sol = total_amount
        req.save(update_fields=["total_recipients", "total_amount_sol"])

    messages.success(request, f"Payout request synced: {total_recipients} recipients.")
    return redirect("creator_campaign_manage", campaign_id=campaign.id)


# ============================================================
# Leaderboard
# ============================================================
@require_http_methods(["GET"])
def leaderboard(request):

    wallet = request.session.get("wallet_address")
    me = None
    if wallet:
        me = UserProfile.objects.filter(wallet_address=wallet).first()

    qs = UserProfile.objects.all().order_by("-points", "created_at")

    paginator = Paginator(qs, 50)
    page = request.GET.get("page", 1)
    rows = paginator.get_page(page)

    # 내 순위 (optional)
    my_rank = None
    if me:
        # 동점 처리: points가 더 크거나, points 동점이면 created_at 더 빠른 사람이 앞
        my_rank = (
            UserProfile.objects
            .filter(points__gt=me.points)
            .count()
            + UserProfile.objects.filter(points=me.points, created_at__lt=me.created_at).count()
            + 1
        )

    return render_m_pc(request, "market/leaderboard", {
        "wallet_address": wallet,
        "me": me,
        "my_rank": my_rank,
        "rows": rows,
    })

# ============================================================
# Public Profile (by wallet)
# ============================================================
@require_http_methods(["GET"])
def public_profile(request, wallet):
    gate = _require_login(request)
    if gate:
        return gate

    target = get_object_or_404(UserProfile, wallet_address=wallet)

    return render(request, "market/public_profile.html", {
        "target": target,
        "wallet_address": request.session.get("wallet_address"),
    })


@require_GET
def announcement_list(request):
    qs = Announcement.objects.filter(is_published=True).order_by("-pinned", "-published_at", "-created_at")
    paginator = Paginator(qs, 10)
    page = request.GET.get("page", 1)
    rows = paginator.get_page(page)

    return render(request, "market/announcements/list.html", {
        "rows": rows,
    })


@require_GET
def announcement_detail(request, slug: str):
    obj = get_object_or_404(Announcement, slug=slug, is_published=True)

    return render(request, "market/announcements/detail.html", {
        "a": obj,
    })


# =====================================================
# Prediction+ section
# =====================================================
OPENAI_API_KEY = settings.OPENAI_API_KEY
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# =====================================================
# 1) Bitget OHLCV
# =====================================================
def fetch_bitget_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=300, market_type="swap"):
    ex = ccxt.bitget({
        "enableRateLimit": True,
        "options": {"defaultType": market_type}  # "spot" or "swap"
    })
    ex.load_markets()

    used_symbol = symbol

    if market_type == "swap" and used_symbol not in ex.markets:
        alt = f"{symbol}:USDT"  # e.g., BTC/USDT:USDT
        if alt in ex.markets:
            used_symbol = alt
        else:
            base = symbol.split("/")[0]
            candidates = [k for k in ex.markets.keys() if base in k and "USDT" in k]
            if not candidates:
                raise ValueError("No suitable market found on Bitget.")
            used_symbol = candidates[0]

    ohlcv = ex.fetch_ohlcv(used_symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df, used_symbol

# =====================================================
# 2) Features
# =====================================================
def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["ret_1"]  = d["close"].pct_change()
    d["ret_3"]  = d["close"].pct_change(3)
    d["ret_12"] = d["close"].pct_change(12)

    d["ma_20"] = d["close"].rolling(20).mean()
    d["ma_50"] = d["close"].rolling(50).mean()
    d["ma_ratio_20"] = d["close"] / d["ma_20"]
    d["ma_ratio_50"] = d["close"] / d["ma_50"]

    d["vol_20"] = d["ret_1"].rolling(20).std()

    # RSI(14)
    delta = d["close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain, index=d.index).rolling(14).mean()
    avg_loss = pd.Series(loss, index=d.index).rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    d["rsi_14"] = 100 - (100 / (1 + rs))

    return d

# =====================================================
# 3) Web Search (TEXT ONLY)
# =====================================================
def get_latest_asset_context(asset: str, hours: int = 24):
    prompt = f"""
Search the web for the most important news/catalysts related to {asset} in the last {hours} hours.
Return 5–8 bullet points. Facts only. No speculation.
Focus on: ETF/flows (if applicable), regulation, security incidents, major exchange/protocol issues, macro news affecting crypto.
"""
    resp = client.responses.create(
        model="gpt-5.2",
        input=prompt,
        tools=[{"type": "web_search"}],
    )
    return resp.output_text.strip()

# =====================================================
# 4) Prediction (JSON MODE)
# =====================================================
def predict_next_hour_direction_json(df_feat: pd.DataFrame, base_asset: str, venue_hint: str, web_context: str = ""):
    d = df_feat.dropna().copy()
    if len(d) < 80:
        raise ValueError("Not enough data after features. Increase limit.")

    recent = d.tail(120)
    latest = recent.iloc[-1]

    payload = {
        "asset": base_asset,
        "venue": venue_hint,
        "now_utc": datetime.now(dt_timezone.utc).isoformat(),
        "last_candle_utc": str(latest["ts"]),
        "last_close": float(latest["close"]),
        "indicators": {
            "ret_1": float(latest["ret_1"]),
            "ret_3": float(latest["ret_3"]),
            "ret_12": float(latest["ret_12"]),
            "ma_ratio_20": float(latest["ma_ratio_20"]),
            "ma_ratio_50": float(latest["ma_ratio_50"]),
            "vol_20": float(latest["vol_20"]),
            "rsi_14": float(latest["rsi_14"]),
        },
        "ohlcv_tail": [
            {
                "ts": str(r.ts),
                "open": float(r.open),
                "high": float(r.high),
                "low": float(r.low),
                "close": float(r.close),
                "volume": float(r.volume),
            }
            for r in recent.tail(60).itertuples(index=False)
        ],
        "web_context": web_context
    }

    instructions = f"""
    You are a professional crypto market analyst.
    
    Task:
    Predict whether the price of {base_asset} will go UP or DOWN over the NEXT 1 hour.
    
    You MUST return valid json only (no markdown, no extra text).
    The json schema is exactly:
    {{
      "verdict": "up" | "down",
      "reasoning": "concise explanation grounded in OHLCV/indicators and relevant news"
    }}
    
    Rules:
    - verdict must be lowercase.
    - include only the two keys shown above.
    - use web_context only if relevant to a 1-hour horizon.
    """

    user_input = "Return json per the schema. Data payload:\n" + json.dumps(payload)

    resp = client.responses.create(
        model="gpt-5.2",
        instructions=instructions,
        input=user_input,
        text={"format": {"type": "json_object"}},
    )
    return json.loads(resp.output_text)

# =====================================================
# 5) One-shot Runner
# =====================================================
def run_pipeline(symbol="BTC/USDT", timeframe="1h", limit=300, market_type="swap", use_web_search=True, web_hours=24):
    base_asset = symbol.split("/")[0]

    df, used_symbol = fetch_bitget_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit, market_type=market_type)
    df_feat = add_basic_features(df)

    web_ctx = ""
    if use_web_search:
        web_ctx = get_latest_asset_context(base_asset, hours=web_hours)

    result = predict_next_hour_direction_json(
        df_feat=df_feat,
        base_asset=base_asset,
        venue_hint=f"Bitget({market_type}:{used_symbol})",
        web_context=web_ctx
    )
    return result

# =========================
# ✅ Page
# =========================
def prediction_plus(request):
    posts = MarketStudyPost.objects.filter(is_published=True).order_by("-created_at")

    return render(request, "market/prediction_plus.html", {
        "market_study_posts": posts,
        "DEV_WALLET": getattr(settings, "LINKHASH_DEV_WALLET", "GkWMcnSNQP4aTVZruJMSwjhzW3VvEza3iZXi3NEwzPKE"),
    })

# =========================
# ✅ API (hourly cached)
# =========================
@require_GET
def api_hourly_predictions(request):
    """
    Returns BTC/ETH/SOL hourly predictions from the DB table.
    Celery updates the table at the top of each hour.
    """
    assets = ["BTC", "ETH", "SOL"]

    now_utc = datetime.now(dt_timezone.utc)
    hour_start = now_utc.replace(minute=0, second=0, microsecond=0)
    next_hour = hour_start + timedelta(hours=1)

    results = {}
    errors = {}
    as_of = None
    next_update = None

    for a in assets:
        try:
            pred = HourlyCryptoPrediction.objects.get(asset=a)
            results[a] = {
                "verdict": pred.verdict,
                "reasoning": pred.reasoning,
            }
            if pred.as_of_utc and (as_of is None or pred.as_of_utc > as_of):
                as_of = pred.as_of_utc
            if pred.next_update_utc and (next_update is None or pred.next_update_utc > next_update):
                next_update = pred.next_update_utc
        except HourlyCryptoPrediction.DoesNotExist:
            errors[a] = "No prediction available yet."

    return JsonResponse({
        "ok": True,
        "as_of_utc": as_of.isoformat() if as_of else hour_start.isoformat(),
        "next_update_utc": next_update.isoformat() if next_update else next_hour.isoformat(),
        "data": results,
        "errors": errors,
    })

def _has_active_market_study_subscription(wallet: str) -> bool:
    """
    wallet이 유효한 'Market Study' 구독을 가지고 있는지 (expires_at > now, verified) 확인.
    """
    wallet = (wallet or "").strip()
    if not wallet:
        return False

    now = timezone.now()

    # ✅ 캐시(선택): DB hit 줄이기
    cache_key = f"lhx_ms_sub_active:{wallet}"
    cached = cache.get(cache_key)
    if cached is not None:
        return bool(cached)

    exists = MarketStudySubscription.objects.filter(
        wallet_address=wallet,
        status=MarketStudySubscription.STATUS_VERIFIED,
        expires_at__gt=now,
    ).exists()

    cache.set(cache_key, 1 if exists else 0, timeout=60)  # 60초만 캐시
    return exists

@require_POST
def api_market_study_subscribe(request):
    """
    POST:
      - asset: "sol" | "lhx"
      - tx_signature: string

    성공 시:
      {
        ok: true,
        expires_at_utc: "YYYY-MM-DD HH:MM:SS UTC",
        expires_at_iso: "<iso utc>"
      }
    """
    gate = _require_login(request)
    if gate:
        return gate

    wallet = (request.session.get("wallet_address") or "").strip()
    if not wallet:
        return JsonResponse({"ok": False, "error": "Missing wallet session."}, status=401)

    asset = (request.POST.get("asset") or "").strip().lower()
    tx_signature = (request.POST.get("tx_signature") or "").strip()

    if asset not in ("sol", "lhx"):
        return JsonResponse({"ok": False, "error": "Invalid asset (sol/lhx)."}, status=400)
    if not tx_signature:
        return JsonResponse({"ok": False, "error": "Missing tx_signature."}, status=400)

    DEV_WALLET = getattr(settings, "LINKHASH_DEV_WALLET", "") or ""
    if not DEV_WALLET:
        return JsonResponse({"ok": False, "error": "Server misconfigured: missing LINKHASH_DEV_WALLET"}, status=500)

    # 가격/기간
    PRICE_SOL = Decimal(str(getattr(settings, "MARKET_STUDY_PRICE_SOL", "0.3")))  # 0.3 SOL
    PRICE_LHX = int(getattr(settings, "MARKET_STUDY_PRICE_LHX", 10_000))        # 10,000 LHX
    SUB_DAYS = int(getattr(settings, "MARKET_STUDY_SUB_DAYS", 30))               # 30 days

    # 중복 tx 방지 (DB unique도 권장)
    if MarketStudySubscription.objects.filter(tx_signature=tx_signature).exists():
        return JsonResponse({"ok": False, "error": "This transaction was already used."}, status=400)

    # ====== On-chain verify ======
    try:
        if asset == "lhx":
            LHX_DECIMALS = int(getattr(settings, "LINKHASH_LHX_DECIMALS", 6))
            required_amount_base = PRICE_LHX * (10 ** LHX_DECIMALS)

            verified = verify_lhx_payment(
                signature=tx_signature,
                payer_wallet=wallet,
                recipient_wallet=DEV_WALLET,
                required_amount_base=required_amount_base,
            )
            if not verified.get("ok"):
                return JsonResponse({"ok": False, "error": verified.get("error", "Payment not verified")}, status=400)

            paid_base = int(verified.get("amount_raw") or 0)
            paid_lhx_int = paid_base // (10 ** LHX_DECIMALS)
            if paid_lhx_int < PRICE_LHX:
                return JsonResponse({"ok": False, "error": "Insufficient LHX amount."}, status=400)

            amount_label = str(paid_lhx_int)

        else:
            # SOL (lamports)
            required_lamports = int((PRICE_SOL * Decimal("1000000000")).to_integral_value())

            verified = verify_sol_payment(
                signature=tx_signature,
                payer_wallet=wallet,
                recipient_wallet=DEV_WALLET,
                required_lamports=required_lamports,
            )
            if not verified.get("ok"):
                return JsonResponse({"ok": False, "error": verified.get("error", "Payment not verified")}, status=400)

            paid_lamports = int(verified.get("lamports") or 0)
            paid_sol = Decimal(paid_lamports) / Decimal("1000000000")
            if paid_sol < PRICE_SOL:
                return JsonResponse({"ok": False, "error": "Insufficient SOL amount."}, status=400)

            amount_label = str(paid_sol)

    except Exception as e:
        return JsonResponse({"ok": False, "error": f"Verification error: {str(e)}"}, status=400)

    # ====== Create subscription ======
    now = timezone.now()

    # 만약 이미 활성 구독이 있으면, "연장" 정책을 추천:
    # - 남은 기간이 있으면 expires_at 기준으로 +30일
    # - 없으면 now 기준으로 +30일
    existing_active = MarketStudySubscription.objects.filter(
        wallet_address=wallet,
        status=MarketStudySubscription.STATUS_VERIFIED,
        expires_at__gt=now,
    ).order_by("-expires_at").first()

    base_time = existing_active.expires_at if existing_active else now
    expires_at = base_time + timedelta(days=SUB_DAYS)

    try:
        with transaction.atomic():
            MarketStudySubscription.objects.create(
                wallet_address=wallet,
                asset=asset,
                amount=amount_label,
                tx_signature=tx_signature,
                status=MarketStudySubscription.STATUS_VERIFIED,
                verified_at=now,
                expires_at=expires_at,
            )
    except IntegrityError:
        return JsonResponse({"ok": False, "error": "Duplicate submission."}, status=400)

    # ✅ 활성 구독 캐시 갱신
    cache.set(f"lhx_ms_sub_active:{wallet}", 1, timeout=60)

    expires_utc = expires_at.astimezone(dt_timezone.utc)
    return JsonResponse({
        "ok": True,
        "expires_at_utc": expires_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "expires_at_iso": expires_utc.isoformat(),
    })

@require_GET
def market_study_detail(request, slug: str):
    """
    Market Study 게시글 상세.
    - 무료글(첫 번째 글)만 비로그인/비구독 접근 허용
    - 나머지는 로그인 + 구독 필요
    - URL 직접 접근도 서버에서 차단
    """
    # ✅ 게시글 조회 (공개된 글만)
    post = get_object_or_404(
        MarketStudyPost,
        slug=slug,
        is_published=True,
    )

    # ✅ 무료글이면 그대로 보여줌
    # (권장: 모델에 is_free boolean 필드 두기)
    if getattr(post, "is_free", False):
        return render(request, "market/market_study/detail.html", {
            "post": post,
            "is_locked": False,
        })

    # ====== 유료글: 로그인 필요 ======
    gate = _require_login(request)
    if gate:
        # 로그인 안 되어 있으면 home으로 보내는 네 정책 그대로
        messages.error(request, "Login required to access premium content.")
        return gate

    wallet = (request.session.get("wallet_address") or "").strip()

    # ====== 유료글: 구독 필요 ======
    if not _has_active_market_study_subscription(wallet):
        # ✅ 직접 접근 차단: Prediction+ 내 Market Study 섹션으로 보내기
        # section id는 템플릿에서 맞춰줘 (#market-study)
        messages.error(request, "Premium content. Please subscribe to access this post.")
        qs = urlencode({
            "ms_subscribe": "1",  # ✅ 모달 자동 오픈 트리거
            "ms_slug": slug,  # (선택) 어떤 글을 보려했는지 기록용
        })
        return redirect(f"{reverse('prediction_plus')}?{qs}#market-study-section")


    # ✅ 구독 OK → 상세 페이지
    return render(request, "market/market_study/detail.html", {
        "post": post,
        "is_locked": False,
        "subscription_active": True,
    })



###########################
# ✅ GAMES SECTION
###########################

# market/views.py

def games_home(request):
    """
    Landing page for Games with FUN Math and Social Proof.
    """

    def format_hms(secs: int) -> str:
        secs = max(0, int(secs))
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def get_timer_for_session(session: GameSession) -> str:
        now = timezone.now()
        secs = 0
        if session.fomo_ends_at:
            secs = (session.fomo_ends_at - now).total_seconds()
        elif session.lottery_draw_at:
            secs = (session.lottery_draw_at - now).total_seconds()
        return format_hms(int(max(0, secs)))

    # ==========================
    # 1) LOTTERY (DB-backed)
    # ==========================
    lottery_ongoing = []
    lottery_past_hourly = []
    lottery_past_daily = []

    lottery_games = Game.objects.filter(category=Game.TYPE_LOTTERY, is_active=True).order_by("id")
    for g in lottery_games:
        session = (
            GameSession.objects.filter(game=g, status=GameSession.STATUS_OPEN)
            .order_by("-created_at")
            .first()
        ) or (
            GameSession.objects.filter(game=g)
            .order_by("-created_at")
            .first()
        )

        if not session:
            continue

        pot = session.pot_total_sol or Decimal("0")
        timer = get_timer_for_session(session)

        # If already settled, treat as past; else ongoing.
        if session.status in (GameSession.STATUS_SETTLED, GameSession.STATUS_CANCELED):
            row = {
                "winner": "Rollover" if not session.winning_numbers else "Winner",
                "nums": session.winning_numbers or "No Match",
                "pot": float(pot),
                "ended": "—",
            }
            if g.subtype == Game.LOTTERY_HOURLY:
                lottery_past_hourly.append(row)
            else:
                lottery_past_daily.append(row)
            continue

        lottery_ongoing.append({
            "id": g.id,  # ✅ real DB id
            "type": "Hourly" if g.subtype == Game.LOTTERY_HOURLY else "Daily",
            "tag": "Live",
            "name": g.name,
            "desc": g.description or "",
            "pot": float(pot),
            "rollover": 0,
            "timer": timer,
            "winning_numbers": [],
            "win_chance": "",
            "ticket_price": float(g.fixed_bet_sol or 0),
            "participants": Participation.objects.filter(session=session).count(),
            "status": "active",
        })

    # ==========================
    # 2) FOMO (Perpetual sessions)
    # ==========================
    fomo_ongoing = []

    # ✅ Define fomo_games BEFORE any loops that use it
    fomo_games = Game.objects.filter(category=Game.TYPE_FOMO, is_active=True).order_by("id")

    def _short_wallet(w: str) -> str:
        w = (w or "").strip()
        if not w:
            return "Unknown"
        # ✅ do NOT shorten "Unknown"
        if w == "Unknown":
            return "Unknown"
        if len(w) <= 12:
            return w
        return f"{w[:4]}...{w[-4:]}"

    # ==========================
    # 2B) FOMO PAST RESULTS (PER GAME, NO "NO-WINNER" ROWS)
    # ==========================
    PAST_ROWS_PER_MODE = 25
    past_rows_by_game = {}  # {game_id: [row, row, ...]}

    for g in fomo_games:
        rows = []

        settled_sessions = (
            GameSession.objects.filter(game=g, status=GameSession.STATUS_SETTLED)
            .select_related("game")
            .order_by("-settled_at", "-created_at")[:PAST_ROWS_PER_MODE]
        )

        for s in settled_sessions:
            win = (
                Winner.objects.filter(session=s)
                .select_related("user_profile", "participation")
                .order_by("-created_at")
                .first()
            )

            # ✅ (2) Skip sessions that have no winner
            if not (win and win.user_profile):
                continue

            winner_wallet = win.user_profile.wallet_address
            winner_disp = _short_wallet(winner_wallet)
            last_bet_amt = float(win.participation.amount_sol) if win.participation else 0.0

            bets = Participation.objects.filter(session=s, status=Participation.STATUS_VERIFIED).count()

            rows.append({
                "winner": winner_disp,
                "bets": bets,
                "last_bet": last_bet_amt,
                "pot": float(s.pot_total_sol or Decimal("0")),
                "ended": s.settled_at.strftime("%Y-%m-%d %H:%M:%S") if s.settled_at else "—",
            })

        past_rows_by_game[g.id] = rows

    # --- chart helpers (10-min buckets) ---
    from datetime import timezone as py_timezone


    def _floor_to_bucket(dt, minutes=10):
        # Django 6: django.utils.timezone has NO timezone.UTC
        # Use Python stdlib timezone: datetime.timezone.utc
        if timezone.is_aware(dt):
            dt = dt.astimezone(py_timezone.utc)
        else:
            dt = dt.replace(tzinfo=py_timezone.utc)

        return dt.replace(minute=(dt.minute // minutes) * minutes, second=0, microsecond=0)

    def _build_pot_series_10m(session: GameSession, bars: int = 12, minutes: int = 10):
        """
        Returns cumulative pot totals at each 10-min bucket end across a rolling window.
        Example: bars=12 -> last 120 minutes.
        """
        now = timezone.now()
        end_bucket = _floor_to_bucket(now, minutes=minutes)
        start_bucket = end_bucket - timezone.timedelta(minutes=minutes * bars)

        # amounts by bucket (verified only)
        qs = (
            Participation.objects.filter(
                session=session,
                status=Participation.STATUS_VERIFIED,
                verified_at__isnull=False,
                verified_at__gte=start_bucket,
            )
            .values("verified_at", "amount_sol")
            .order_by("verified_at")
        )

        # sum amounts per bucket-start
        bucket_sums = {}
        for row in qs:
            b = _floor_to_bucket(row["verified_at"], minutes=minutes)
            bucket_sums[b] = bucket_sums.get(b, Decimal("0")) + (row["amount_sol"] or Decimal("0"))

        # produce cumulative totals across buckets
        series = []
        running = Decimal("0")
        for i in range(bars):
            b = start_bucket + timezone.timedelta(minutes=minutes * i)
            running += bucket_sums.get(b, Decimal("0"))
            series.append(float(running))

        return series

    for g in fomo_games:
        # ✅ guarantee an OPEN session exists, and auto-settle if expired
        session = fomo_engine._ensure_current_fomo_session(g)

        pot = session.pot_total_sol or Decimal("0")
        timer = get_timer_for_session(session)

        # ✅ define bets_count for template (verified only)
        bets_count = Participation.objects.filter(
            session=session,
            status=Participation.STATUS_VERIFIED
        ).count()

        # King = latest VERIFIED bet (fallback Unknown)
        # ✅ For "limit" mode, king is determined by block_time
        king_part = fomo_engine._get_king_participation(g, session)
        king_wallet = king_part.user_profile.wallet_address if king_part else "Unknown"

        # last bet (for UI) — same as king
        prev_bid = float(king_part.amount_sol) if king_part else 0.0

        # ✅ next bid shown in UI should match server rule
        required_next = fomo_engine._compute_required_bet_sol(g, session)

        is_limit_mode = (g.subtype == "limit")

        # ✅ 10-min bars (last 120 minutes -> 12 bars)
        pot_series = _build_pot_series_10m(session, bars=12, minutes=10)
        series_max = max(pot_series) if pot_series else 0.0
        if series_max <= 0:
            series_max = 1.0  # avoid divide-by-zero in template

        # subtype values: "limit" or "no_limit"
        is_limit_mode = (g.subtype == "limit")

        fomo_ongoing.append({
            "id": g.id,
            "subtype": g.subtype,  # ✅ keep raw subtype
            "type": "Limit" if is_limit_mode else "No-Limit",
            "tag": "Fixed Entry" if is_limit_mode else "High Stakes",
            "name": g.name,
            "desc": g.description or "",
            "pot": float(pot),
            "timer": timer,

            # ✅ (3) provide ends_at for client countdown (real reset on WS)
            "ends_at": session.fomo_ends_at.isoformat() if session.fomo_ends_at else None,

            "king": king_wallet,

            # ✅ (4) server-truth required next bet streamed to UI
            "next_bid": float(required_next),
            "prev_bid": prev_bid,
            "bets_count": bets_count,
            "limit_mode": is_limit_mode,

            # ✅ (1) aligned past rows per game
            "past_rows": past_rows_by_game.get(g.id, []),

            # ✅ chart payload for template
            "pot_series": pot_series,
            "pot_series_max": float(series_max),
        })




    # ==========================
    # 3) USER HUD (REAL)
    # ==========================
    user_active_bets = []
    user_past_history = []

    wallet = (request.session.get("wallet_address") or "").strip()
    if wallet:
        prof = UserProfile.objects.filter(wallet_address=wallet).first()
        if prof:
            # --------
            # Live Bets (FOMO only for now)
            # One row per open session where user participated (latest participation)
            # --------
            live_parts = (
                Participation.objects.filter(
                    user_profile=prof,
                    session__status=GameSession.STATUS_OPEN,
                    session__game__category=Game.TYPE_FOMO,
                )
                .select_related("session", "session__game")
                .order_by("session_id", "-created_at")
            )

            seen_sessions = set()
            for p in live_parts:
                sid = p.session_id
                if sid in seen_sessions:
                    continue
                seen_sessions.add(sid)

                s = p.session
                g = s.game

                # Current king for that session
                # ✅ For "limit" mode, king is determined by block_time
                king_part = fomo_engine._get_king_participation(g, s)
                king_wallet = king_part.user_profile.wallet_address if king_part else "Unknown"
                status = "Live"
                if king_part and king_part.user_profile_id == prof.id:
                    status = "Winning"

                # Timer seconds (string expected by your HUD markup; it reads innerText)
                now = timezone.now()
                secs = 0
                if s.fomo_ends_at:
                    secs = int(max(0, (s.fomo_ends_at - now).total_seconds()))
                timer_str = fomo_engine._format_hms(secs)

                user_active_bets.append({
                    "game_id": g.id,  # ✅ needed for WS updates
                    "game_name": g.name,
                    "details": f"Round #{s.round_no} · {('Limit' if g.subtype=='limit' else 'No-Limit')}",
                    "status": status,
                    "timer": timer_str,
                    "ends_at": s.fomo_ends_at.isoformat() if s.fomo_ends_at else None,  # ✅ WS timer sync
                    "wager": float(p.amount_sol),
                    "your_pick": "-",
                })


            # --------
            # History (FOMO only for now)
            # Show last N settled sessions where user participated
            # --------
            past_sessions = (
                GameSession.objects.filter(
                    participations__user_profile=prof,
                    status=GameSession.STATUS_SETTLED,
                    game__category=Game.TYPE_FOMO,
                )
                .select_related("game")
                .distinct()
                .order_by("-settled_at", "-created_at")[:50]
            )

            for s in past_sessions:
                g = s.game
                spent = (
                    Participation.objects.filter(session=s, user_profile=prof, status=Participation.STATUS_VERIFIED)
                    .aggregate(total=models.Sum("amount_sol"))
                    .get("total") or Decimal("0")
                )

                win = Winner.objects.filter(session=s, user_profile=prof).first()
                if win:
                    outcome = "win"
                    # ✅ show payout (always positive) for WON rows (hard guard)
                    payout = (win.payout_amount_sol or Decimal("0")).quantize(Decimal("0.000000001"))
                    payout = abs(payout)
                    winner_wallet = prof.wallet_address

                    profit_display = f"+{payout}"
                    profit_num = payout

                else:
                    outcome = "loss"
                    wrow = Winner.objects.filter(session=s).select_related("user_profile").first()
                    winner_wallet = wrow.user_profile.wallet_address if (wrow and wrow.user_profile) else "Unknown"

                    # loss stays as-is (your template uses -wager for loss rows)
                    profit_display = "+0"
                    profit_num = Decimal("0")


                user_past_history.append({
                    "type": "FOMO",
                    "game_name": g.name,
                    "date": (s.settled_at.strftime("%Y-%m-%d %H:%M:%S") if s.settled_at else "—"),
                    "outcome": outcome,
                    "profit_display": profit_display,   # ✅ win shows +payout now
                    "profit": float(profit_num),        # keep numeric if you need it later
                    "wager": float(spent),
                    "winner": winner_wallet,
                    "your_pick": "-",
                    "winning_pick": "-",
                })



    # ✅ Solana wallet registration (for bet-gating + sender validation)
    sol_wallets = []
    primary_sol_wallet = ""
    has_solana_wallet = False

    if wallet:
        prof = UserProfile.objects.filter(wallet_address=wallet).first()
        if prof:
            # NOTE: UserSolanaWallet model added in models.py
            qs = prof.solana_wallets.all().order_by("-is_primary", "-updated_at")
            sol_wallets = [w.solana_wallet for w in qs]
            primary = qs.filter(is_primary=True).first()
            primary_sol_wallet = primary.solana_wallet if primary else (sol_wallets[0] if sol_wallets else "")
            has_solana_wallet = bool(sol_wallets)

    return render(request, "market/games/home.html", {
        "lottery_ongoing": lottery_ongoing,
        "lottery_past_hourly": lottery_past_hourly,
        "lottery_past_daily": lottery_past_daily,

        # ✅ per-game aligned past rows are inside each game object:
        # game.past_rows is used in the template include
        "fomo_ongoing": fomo_ongoing,

        "user_active_bets": user_active_bets,
        "user_past_history": user_past_history,

        # ✅ needed by HUD + websocket UI comparisons
        "wallet_address": wallet,

        # ✅ NEW: bet gate + UI
        "has_solana_wallet": has_solana_wallet,
        "primary_solana_wallet": primary_sol_wallet,
        "solana_wallets": sol_wallets,
    })






def lottery_detail(request, game_id):
    gate = _require_login(request)
    
    # Mock Detail
    game = {
        "id": game_id,
        "name": "Hourly Jackpot #492" if game_id == 101 else "Daily Grand #55",
        "type": "hourly" if game_id == 101 else "daily",
        "pot": 12.5 if game_id == 101 else 145.8,
        "price": 0.1 if game_id == 101 else 1.0,
        "timer": "00:42:15",
        "wallet": "GkWMcnSNQP4aTVZruJMSwjhzW3VvEza3iZXi3NEwzPKE", # Game wallet
        "win_chance": "10%" if game_id == 101 else "1%",
        "history": [
             {"block": 284100, "user": "UserA", "nums": [1,4,9]},
             {"block": 284105, "user": "UserB", "nums": [2,5,8]},
        ]
    }

    if request.method == "POST":
        if gate: return gate # Login required to bet
        
        tx_id = request.POST.get("tx_signature")
        # Logic to save bet to DB...
        from django.contrib import messages
        messages.success(request, "Ticket verified! Good luck.")
        return redirect("games_home")

    return render(request, "market/games/lottery_detail.html", {"game": game, "wallet_address": request.session.get("wallet_address")})

def fomo_detail(request, game_id):
    gate = _require_login(request)

    def format_hms(secs: int) -> str:
        secs = max(0, int(secs))
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    game_obj = get_object_or_404(Game, pk=game_id, is_active=True, category=Game.TYPE_FOMO)

    session = (
        GameSession.objects.filter(game=game_obj, status=GameSession.STATUS_OPEN)
        .order_by("-created_at")
        .first()
    ) or (
        GameSession.objects.filter(game=game_obj)
        .order_by("-created_at")
        .first()
    )

    if not session:
        # No session yet: still render page, but show safe defaults.
        timer_str = "00:00:00"
        pot_sol = Decimal("0")
        last_bidder = "Unknown"
        history_rows = []
    else:
        now = timezone.now()
        timer_secs = 0
        if session.fomo_ends_at:
            timer_secs = int(max(0, (session.fomo_ends_at - now).total_seconds()))
        timer_str = format_hms(timer_secs)

        pot_sol = session.pot_total_sol or Decimal("0")

        # last bidder = latest VERIFIED bet (fallback: latest PENDING bet, then Unknown)
        last_verified = (
            Participation.objects.filter(session=session, status=Participation.STATUS_VERIFIED)
            .order_by("-bet_index", "-created_at")
            .select_related("user_profile")
            .first()
        )
        last_any = (
            Participation.objects.filter(session=session)
            .order_by("-bet_index", "-created_at")
            .select_related("user_profile")
            .first()
        )

        last_bidder = "Unknown"
        if last_verified and last_verified.user_profile:
            last_bidder = last_verified.user_profile.wallet_address
        elif last_any and last_any.user_profile:
            last_bidder = last_any.user_profile.wallet_address

        # history: show recent participations (verified first, then pending)
        recent = (
            Participation.objects.filter(session=session)
            .order_by("-created_at")[:25]
            .select_related("user_profile")
        )
        history_rows = []
        for p in recent:
            history_rows.append({
                "time": p.created_at.strftime("%H:%M:%S"),
                "user": p.user_profile.wallet_address if p.user_profile else "Unknown",
                "amt": float(p.amount_sol),
            })

    # limit_mode string for your template
    limit_mode_str = "Limit" if (game_obj.subtype == "limit") else "No-Limit"

    # wallet/min_bet/fixed bet derived from game table
    # (template uses min_bet only in No-Limit branch; your modal uses both)
    fixed_bet = game_obj.fixed_bet_sol or Decimal("0")
    min_bet = game_obj.min_bet_sol or Decimal("0")

    game = {
        "id": game_obj.id,
        "name": game_obj.name,  # ✅ keep name
        "pot": float(pot_sol),
        "timer": timer_str,
        "last_bidder": last_bidder,
        "limit_mode": limit_mode_str,
        "wallet": game_obj.sol_public_key,
        "min_bet": float(fixed_bet if fixed_bet > 0 else min_bet),
        "history": history_rows,  # ✅ keep history (real now)
    }

    if request.method == "POST":
        if gate:
            return gate

        tx_id = (request.POST.get("tx_signature") or "").strip()
        amount = (request.POST.get("amount") or "").strip()

        # MVP behavior: just accept POST and redirect for now (you already have modal bet API for real)
        from django.contrib import messages
        messages.success(request, "Bet submitted. Pending verification.")
        return redirect("games_home")

    return render(
        request,
        "market/games/fomo_detail.html",
        {"game": game, "wallet_address": request.session.get("wallet_address")}
    )

def _solana_get_transaction(signature: str) -> dict | None:
    """
    Fetch parsed transaction from Solana RPC.
    Returns dict or None if not found.
    """
    url = settings.SOLANA_RPC_URL
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTransaction",
        "params": [
            signature,
            {
                "encoding": "jsonParsed",
                "maxSupportedTransactionVersion": 0,
                "commitment": "confirmed",
            }
        ],
    }
    try:
        r = requests.post(url, json=payload, timeout=12)
        r.raise_for_status()
        j = r.json()
        return j.get("result")
    except Exception:
        return None


def _extract_sol_transfer_to_receiver(tx: dict, receiver_wallet: str) -> tuple[Decimal | None, int | None, set[str]]:
    """
    Extract total SOL transferred to receiver_wallet from system transfer instructions.
    Also collects possible sender wallet(s) from the parsed 'source' field.

    Returns:
      (amount_sol_or_None, block_time_unix_or_None, senders_set)

    If not found:
      (None, block_time, senders_set)
    """
    if not tx:
        return (None, None, set())

    block_time = tx.get("blockTime")  # unix seconds or None

    meta = tx.get("meta") or {}
    if meta.get("err") is not None:
        return (None, block_time, set())

    trx = tx.get("transaction") or {}
    msg = trx.get("message") or {}
    instructions = msg.get("instructions") or []

    total_lamports = 0
    senders: set[str] = set()

    for ix in instructions:
        # jsonParsed format:
        # ix = {"program":"system","parsed":{"type":"transfer","info":{"source":"...","destination":"...","lamports":123}}}
        prog = ix.get("program")
        parsed = ix.get("parsed") or {}
        if prog != "system":
            continue
        if parsed.get("type") != "transfer":
            continue
        info = parsed.get("info") or {}
        if info.get("destination") != receiver_wallet:
            continue

        src = (info.get("source") or "").strip()
        if src:
            senders.add(src)

        lamports = info.get("lamports")
        if isinstance(lamports, int):
            total_lamports += lamports

    if total_lamports <= 0:
        return (None, block_time, senders)

    sol = (Decimal(total_lamports) / Decimal(1_000_000_000)).quantize(Decimal("0.000000001"))
    return (sol, block_time, senders)



@require_GET
def game_session_state_json(request, game_id: int):
    """
    Returns the latest OPEN (or most recent) session data for a Game.
    Used by the bet modal to display live info.
    """
    game = get_object_or_404(Game, pk=game_id, is_active=True)

    # ✅ Perpetual sessions (FOMO): auto-create and auto-settle on request
    if game.category == Game.TYPE_FOMO:
        session = fomo_engine._ensure_current_fomo_session(game)
    else:
        # Lottery logic stays as-is for now (you can expand later)
        session = (
            GameSession.objects.filter(game=game, status=GameSession.STATUS_OPEN)
            .order_by("-created_at")
            .first()
        ) or (
            GameSession.objects.filter(game=game)
            .order_by("-created_at")
            .first()
        )

        if not session:
            return JsonResponse({
                "ok": False,
                "error": "No session found for this game yet. Create a GameSession in admin.",
            }, status=404)

    # King = latest verified bet/ticket (for FOMO, kind=bet is typical)
    # ✅ For "limit" mode, king is determined by block_time
    king_part = fomo_engine._get_king_participation(game, session)
    king_wallet = king_part.user_profile.wallet_address if king_part else "Unknown"

    # ✅ prev bid (from king)
    prev_bid_sol = str(king_part.amount_sol) if king_part and king_part.amount_sol is not None else "0"
    # ✅ NEW: social proof count (VERIFIED only)
    bets_count = Participation.objects.filter(
        session=session,
        status=Participation.STATUS_VERIFIED
    ).count()

    # Timer: for FOMO use fomo_ends_at, else for lottery use lottery_draw_at
    now = timezone.now()
    timer_secs = 0
    if session.fomo_ends_at:
        timer_secs = int(max(0, (session.fomo_ends_at - now).total_seconds()))
    elif session.lottery_draw_at:
        timer_secs = int(max(0, (session.lottery_draw_at - now).total_seconds()))

    def format_hms(secs: int) -> str:
        h = secs // 3600
        m = (secs % 3600) // 60
        s = secs % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    required_bet = fomo_engine._compute_required_bet_sol(game, session)

    # ✅ NEW: force increment display rule for UI
    inc_ui = "0" if (game.subtype == "limit") else "3"

    return JsonResponse({
        "ok": True,
        "game": {
            "id": game.id,
            "name": game.name,
            "category": game.category,
            "subtype": game.subtype,
            "protocol_fee_pct": str(game.protocol_fee_pct),
            "sol_wallet": game.sol_public_key,
            "min_bet_sol": str(game.min_bet_sol),
            "fixed_bet_sol": str(game.fixed_bet_sol),

            # ✅ always show correct rule to UI
            "increment_pct": inc_ui,
        },
        "session": {
            "id": session.id,
            "round_no": session.round_no,
            "status": session.status,
            "pot_total_sol": str(session.pot_total_sol),
            "timer": format_hms(timer_secs),
            "king_wallet": king_wallet,
            "required_bet_sol": str(required_bet),
            "prev_bid_sol": prev_bid_sol,     # ✅ added
            "bets_count": bets_count,         # ✅ added
            "fomo_ends_at": session.fomo_ends_at.isoformat() if session.fomo_ends_at else None,
        }
    })



@require_POST
def game_place_bet(request, game_id: int):
    """
    FOMO mechanics:
    - Ensure perpetual session exists
    - Verify Solana TX synchronously
    - Enforce rules using ON-CHAIN amount + blockTime
    - Store VERIFIED Participation
    - Reset timer and broadcast websocket update
    - Log invalid bets for manual refund
    """
    gate = _require_login(request)
    if gate:
        return gate

    game = get_object_or_404(Game, pk=game_id, is_active=True)

    if game.category != Game.TYPE_FOMO:
        return JsonResponse({"ok": False, "error": "Only FOMO betting is implemented here for now."}, status=400)

    # ✅ Ensure current session exists, and auto-settle expired ones on demand
    session = fomo_engine._ensure_current_fomo_session(game)

    wallet = (request.session.get("wallet_address") or "").strip()
    if not wallet:
        return JsonResponse({"ok": False, "error": "Wallet not connected."}, status=401)

    user, _ = UserProfile.objects.get_or_create(wallet_address=wallet)

    tx = (request.POST.get("tx_signature") or "").strip()
    if not tx:
        return JsonResponse({"ok": False, "error": "Missing tx_signature"}, status=400)

    # Client-sent amount is NOT source of truth; we only use it for nicer errors.
    client_amount_str = (request.POST.get("amount_sol") or "").strip()

    # Prevent duplicate tx
    if Participation.objects.filter(tx_signature=tx).exists():
        InvalidBet.objects.create(
            game=game,
            session=session,
            submitter_wallet=wallet,
            receiver_wallet=game.sol_public_key,
            tx_signature=tx,
            reason=InvalidBet.REASON_DUPLICATE_TX,
            note="TX signature already used in Participation.",
        )
        return JsonResponse({"ok": False, "error": "This transaction was already used."}, status=400)

    now = timezone.now()

    # If the timer already expired at the moment of submission, reject early (still log)
    if session.fomo_ends_at and now >= session.fomo_ends_at:
        InvalidBet.objects.create(
            game=game,
            session=session,
            submitter_wallet=wallet,
            receiver_wallet=game.sol_public_key,
            tx_signature=tx,
            reason=InvalidBet.REASON_LATE,
            note="Submitted after session already expired (server time).",
        )
        # Also auto-settle and move on
        fomo_engine._ensure_current_fomo_session(game)
        return JsonResponse({"ok": False, "error": "Too late. Session already ended."}, status=400)

    # ✅ Fetch and parse transaction
    tx_obj = _solana_get_transaction(tx)
    if not tx_obj:
        InvalidBet.objects.create(
            game=game,
            session=session,
            submitter_wallet=wallet,
            receiver_wallet=game.sol_public_key,
            tx_signature=tx,
            reason=InvalidBet.REASON_TX_NOT_FOUND,
            note="RPC getTransaction returned no result (not found/confirmed yet). Try again in a few seconds.",
        )
        return JsonResponse({"ok": False, "error": "TX not found yet. Wait 5–15 seconds and retry."}, status=400)

    # ✅ Pull user's registered Solana sender wallet(s)
    # NOTE: import at top of file: from .models import UserSolanaWallets
    registered_senders = set(
        (w or "").strip()
        for w in UserSolanaWallet.objects.filter(user_profile=user)
        .values_list("solana_wallet", flat=True)
    )
    registered_senders.discard("")  # safety

    amount_onchain, block_time_unix, sender_set = _extract_sol_transfer_to_receiver(tx_obj, game.sol_public_key)

    if amount_onchain is None:
        # Might be failed tx or wrong receiver
        meta = (tx_obj.get("meta") or {})
        if meta.get("err") is not None:
            reason = InvalidBet.REASON_TX_FAILED
            note = f"meta.err={meta.get('err')}"
        else:
            reason = InvalidBet.REASON_BAD_RECEIVER
            note = "No system transfer to game wallet detected."

        InvalidBet.objects.create(
            game=game,
            session=session,
            submitter_wallet=wallet,
            receiver_wallet=game.sol_public_key,
            tx_signature=tx,
            reason=reason,
            note=note,
        )
        return JsonResponse({"ok": False, "error": "TX does not transfer SOL to the game wallet."}, status=400)

    # Convert blockTime to aware datetime (UTC)
    from datetime import timezone as py_timezone

    block_dt = None
    if isinstance(block_time_unix, int):
        block_dt = timezone.datetime.fromtimestamp(block_time_unix, tz=py_timezone.utc)

    # ✅ Sender validation: TX sender must match one of user's registered Solana wallets
    # If multiple senders exist (rare), accept if ANY matches.
    # If user somehow has no registered_senders, always reject here.
    sender_ok = bool(sender_set) and bool(registered_senders) and bool(sender_set.intersection(registered_senders))
    if not sender_ok:
        InvalidBet.objects.create(
            game=game,
            session=session,
            submitter_wallet=wallet,  # login identity
            receiver_wallet=game.sol_public_key,
            tx_signature=tx,
            amount_onchain_sol=amount_onchain,
            block_time=block_dt,
            reason=InvalidBet.REASON_BAD_SENDER,
            note=f"TX sender(s)={sorted(sender_set)} not in registered={sorted(registered_senders)}",
        )
        return JsonResponse(
            {"ok": False, "error": "Invalid sender. Please send SOL from your registered Solana wallet."},
            status=400,
        )

    # ✅ NEW: For "limit" mode, reject bets with block_time too old (older than 1 timer window)
    is_limit = (game.subtype == "limit")
    if is_limit and block_dt:
        window_seconds = int(game.fomo_timer_seconds or 600)
        too_old_threshold = now - timedelta(seconds=window_seconds)
        if block_dt < too_old_threshold:
            InvalidBet.objects.create(
                game=game,
                session=session,
                submitter_wallet=wallet,
                receiver_wallet=game.sol_public_key,
                tx_signature=tx,
                amount_onchain_sol=amount_onchain,
                block_time=block_dt,
                reason=InvalidBet.REASON_TOO_OLD,
                note=f"Limit mode: block_time {block_dt.isoformat()} older than threshold {too_old_threshold.isoformat()}",
            )
            return JsonResponse({"ok": False, "error": "Transaction is too old. Must be within the last timer window."}, status=400)

    # Enforce: transaction must be BEFORE timer ends
    if session.fomo_ends_at and block_dt and block_dt > session.fomo_ends_at:
        InvalidBet.objects.create(
            game=game,
            session=session,
            submitter_wallet=wallet,
            receiver_wallet=game.sol_public_key,
            tx_signature=tx,
            amount_onchain_sol=amount_onchain,
            block_time=block_dt,
            reason=InvalidBet.REASON_LATE,
            note="TX confirmed after session deadline (blockTime > fomo_ends_at).",
        )
        # Session might already be over now; allow the system to settle/advance
        fomo_engine._ensure_current_fomo_session(game)
        return JsonResponse({"ok": False, "error": "Too late. TX landed after the timer expired."}, status=400)
    # ✅ Enforce bet rules using server truth
    required = fomo_engine._compute_required_bet_sol(game, session)

    last_any = (
        Participation.objects.filter(session=session)
        .order_by("-bet_index", "-created_at")
        .first()
    )
    prev_bid_sol = str(last_any.amount_sol) if last_any and last_any.amount_sol is not None else "0"


    # If fixed bet mode and you want strict equality, enforce equality to fixed (or required)
    is_limit = (game.subtype == "limit")
    if is_limit:
        # strict: must equal required (which is max(0.01, fixed))
        if amount_onchain != required:
            InvalidBet.objects.create(
                game=game,
                session=session,
                submitter_wallet=wallet,
                receiver_wallet=game.sol_public_key,
                tx_signature=tx,
                amount_onchain_sol=amount_onchain,
                block_time=block_dt,
                reason=InvalidBet.REASON_BAD_AMOUNT,
                note=f"Fixed mode: required={required} onchain={amount_onchain} client={client_amount_str}",
            )
            return JsonResponse({"ok": False, "error": f"Fixed bet required: {required} SOL"}, status=400)
    else:
        if amount_onchain < required:
            InvalidBet.objects.create(
                game=game,
                session=session,
                submitter_wallet=wallet,
                receiver_wallet=game.sol_public_key,
                tx_signature=tx,
                amount_onchain_sol=amount_onchain,
                block_time=block_dt,
                reason=InvalidBet.REASON_BAD_AMOUNT,
                note=f"No-limit: required>={required} onchain={amount_onchain} client={client_amount_str}",
            )
            return JsonResponse({"ok": False, "error": f"Minimum bet right now is {required} SOL"}, status=400)

    # ✅ Create VERIFIED participation + update pot + reset timer atomically
    with transaction.atomic():
        # Re-check duplicate within transaction
        if Participation.objects.select_for_update().filter(tx_signature=tx).exists():
            InvalidBet.objects.create(
                game=game,
                session=session,
                submitter_wallet=wallet,
                receiver_wallet=game.sol_public_key,
                tx_signature=tx,
                amount_onchain_sol=amount_onchain,
                block_time=block_dt,
                reason=InvalidBet.REASON_DUPLICATE_TX,
                note="Duplicate detected inside transaction.",
            )
            return JsonResponse({"ok": False, "error": "This transaction was already used."}, status=400)

        # lock session row so required bet / timer doesn’t race
        session = GameSession.objects.select_for_update().get(pk=session.pk)

        # if session expired during race, reject (still log)
        now2 = timezone.now()
        if session.fomo_ends_at and now2 >= session.fomo_ends_at:
            InvalidBet.objects.create(
                game=game,
                session=session,
                submitter_wallet=wallet,
                receiver_wallet=game.sol_public_key,
                tx_signature=tx,
                amount_onchain_sol=amount_onchain,
                block_time=block_dt,
                reason=InvalidBet.REASON_LATE,
                note="Race: session expired during bet submission.",
            )
            return JsonResponse({"ok": False, "error": "Too late. Session ended."}, status=400)

        # bet_index increment
        last_idx = (
            Participation.objects.filter(session=session)
            .aggregate(mx=models.Max("bet_index"))
            .get("mx") or 0
        )

        Participation.objects.create(
            session=session,
            user_profile=user,
            kind=Participation.KIND_BET,
            status=Participation.STATUS_VERIFIED,  # ✅ verified immediately
            amount_sol=amount_onchain,             # ✅ on-chain truth
            tx_signature=tx,
            bet_index=int(last_idx) + 1,
            verified_at=now2,
            block_time=block_dt,                   # ✅ on-chain block timestamp
        )

        session.pot_total_sol = (session.pot_total_sol or Decimal("0")) + amount_onchain
        session.fomo_ends_at = now2 + timezone.timedelta(seconds=int(game.fomo_timer_seconds or 600))
        session.save(update_fields=["pot_total_sol", "fomo_ends_at"])

    # ✅ Broadcast realtime update
    fomo_engine._broadcast_game_update(game, session)

    # ============================================================
    # ✅ Production: dynamically schedule settlement at fomo_ends_at
    # - revoke previous scheduled task (if any)
    # - schedule a new one at the updated ends_at
    # - persist task id to session for future revoke
    # ============================================================
    # Dynamic settlement scheduling (no circular imports)
    session = GameSession.objects.get(pk=session.pk)  # refresh ends_at/task_id
    fomo_engine._schedule_fomo_settlement(session)


    # Return updated state so the bettor's own tab updates instantly (even if WS is delayed)
    # Shape matches game_session_state_json response closely.
    now3 = timezone.now()
    timer_secs = 0
    if session.fomo_ends_at:
        timer_secs = int(max(0, (session.fomo_ends_at - now3).total_seconds()))

    king_part = fomo_engine._get_king_participation(game, session)
    king_wallet = king_part.user_profile.wallet_address if king_part else "Unknown"
    required_bet = fomo_engine._compute_required_bet_sol(game, session)

    return JsonResponse({
        "ok": True,
        "game": {
            "id": game.id,
            "name": game.name,
            "category": game.category,
            "subtype": game.subtype,
            "protocol_fee_pct": str(game.protocol_fee_pct),
            "sol_wallet": game.sol_public_key,
            "min_bet_sol": str(game.min_bet_sol),
            "fixed_bet_sol": str(game.fixed_bet_sol),
            "increment_pct": "0" if is_limit else "3",
        },
        "session": {
            "id": session.id,
            "status": session.status,
            "pot_total_sol": str(session.pot_total_sol),
            "prev_bid_sol": prev_bid_sol,
            "timer": fomo_engine._format_hms(timer_secs),
            "king_wallet": king_wallet,
            "required_bet_sol": str(required_bet),
            "fomo_ends_at": session.fomo_ends_at.isoformat() if session.fomo_ends_at else None,
        }
    })


@require_GET
def user_hud_state_json(request):
    """
    Return the latest HUD state for the connected wallet:
    - live bets (FOMO open sessions the user joined)
    - history (FOMO settled sessions the user joined)
    """
    wallet = (request.session.get("wallet_address") or "").strip()
    if not wallet:
        return JsonResponse({"ok": True, "wallet": "", "active": [], "history": []})

    prof = UserProfile.objects.filter(wallet_address=wallet).first()
    if not prof:
        return JsonResponse({"ok": True, "wallet": wallet, "active": [], "history": []})

    # Live Bets (same logic as games_home, but JSON-friendly)
    active = []
    live_parts = (
        Participation.objects.filter(
            user_profile=prof,
            session__status=GameSession.STATUS_OPEN,
            session__game__category=Game.TYPE_FOMO,
        )
        .select_related("session", "session__game")
        .order_by("session_id", "-created_at")
    )

    seen_sessions = set()
    for p in live_parts:
        sid = p.session_id
        if sid in seen_sessions:
            continue
        seen_sessions.add(sid)

        s = p.session
        g = s.game

        king_part = fomo_engine._get_king_participation(g, s)
        king_wallet = king_part.user_profile.wallet_address if king_part else "Unknown"
        status = "Winning" if (king_part and king_part.user_profile_id == prof.id) else "Live"

        active.append({
            "game_id": g.id,
            "game_name": g.name,
            "details": f"Round #{s.round_no} · {('Limit' if g.subtype=='limit' else 'No-Limit')}",
            "status": status,
            "ends_at": s.fomo_ends_at.isoformat() if s.fomo_ends_at else "",
            "wager": str(p.amount_sol),
        })

    # History (same idea as games_home)
    history = []
    past_sessions = (
        GameSession.objects.filter(
            participations__user_profile=prof,
            status=GameSession.STATUS_SETTLED,
            game__category=Game.TYPE_FOMO,
        )
        .select_related("game")
        .distinct()
        .order_by("-settled_at", "-created_at")[:50]
    )

    for s in past_sessions:
        g = s.game
        spent = (
            Participation.objects.filter(session=s, user_profile=prof, status=Participation.STATUS_VERIFIED)
            .aggregate(total=models.Sum("amount_sol"))
            .get("total") or Decimal("0")
        )

        win = Winner.objects.filter(session=s, user_profile=prof).first()
        if win:
            payout = (win.payout_amount_sol or Decimal("0")).quantize(Decimal("0.000000001"))
            payout = abs(payout)  # ✅ hard guard
            outcome = "win"
            profit_display = f"+{payout}"
            winner_wallet = prof.wallet_address
        else:
            outcome = "loss"
            wrow = Winner.objects.filter(session=s).select_related("user_profile").first()
            winner_wallet = wrow.user_profile.wallet_address if (wrow and wrow.user_profile) else "Unknown"
            profit_display = "+0"

        history.append({
            "game_name": g.name,
            "date": (s.settled_at.strftime("%Y-%m-%d %H:%M:%S") if s.settled_at else "—"),
            "outcome": outcome,
            "profit_display": profit_display,
            "wager": str(spent),
            "winner": winner_wallet,
        })

    return JsonResponse({"ok": True, "wallet": wallet, "active": active, "history": history})


@require_POST
def user_update_solana_wallet(request):
    """
    Logged-in user registers/updates their Solana wallet used for Games.
    - Makes the submitted wallet the PRIMARY wallet
    - Enforces global uniqueness (model: unique=True)
    """
    gate = _require_login(request)
    if gate:
        return gate

    wallet = (request.session.get("wallet_address") or "").strip()
    if not wallet:
        return JsonResponse({"ok": False, "error": "Wallet not connected."}, status=401)

    prof = UserProfile.objects.filter(wallet_address=wallet).first()
    if not prof:
        return JsonResponse({"ok": False, "error": "User profile not found."}, status=404)

    sol = (request.POST.get("solana_wallet") or "").strip()
    if not sol:
        return JsonResponse({"ok": False, "error": "Missing solana_wallet"}, status=400)

    # lightweight validation (Solana base58 addresses are typically 32~44 chars)
    if len(sol) < 32 or len(sol) > 64:
        return JsonResponse({"ok": False, "error": "Invalid Solana wallet length."}, status=400)

    # Make this wallet primary, and ensure only one primary.
    try:
        with transaction.atomic():
            # set others to non-primary
            prof.solana_wallets.select_for_update().update(is_primary=False)

            # upsert this wallet (unique=True ensures global uniqueness)
            obj, created = UserSolanaWallet.objects.get_or_create(
                solana_wallet=sol,
                defaults={"user_profile": prof, "is_primary": True},
            )

            # If it existed but belonged to same user, update it as primary.
            if not created:
                if obj.user_profile_id != prof.id:
                    return JsonResponse(
                        {"ok": False, "error": "This Solana wallet is already registered by another user."},
                        status=400,
                    )
                obj.is_primary = True
                obj.save(update_fields=["is_primary", "updated_at"])

    except Exception as e:
        # Handles race/constraint errors
        return JsonResponse({"ok": False, "error": f"Failed to save Solana wallet: {str(e)[:160]}"}, status=400)

    # Return updated list for UI
    wallets = list(prof.solana_wallets.all().order_by("-is_primary", "-updated_at").values_list("solana_wallet", flat=True))
    return JsonResponse({
        "ok": True,
        "primary": sol,
        "wallets": wallets,
    })
