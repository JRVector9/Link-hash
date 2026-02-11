# market/admin.py
from django.contrib import admin
from django.utils.html import format_html
from django.utils import timezone
from django.db import transaction
import csv
from django.http import HttpResponse
from django import forms
from ckeditor_uploader.widgets import CKEditorUploadingWidget



from market.models import (
    TokenSnapshot,
    SponsoredProject,
    SponsoredAdSubmission,
    UserProfile,
    Campaign,
    CampaignField,
    CampaignSubmission,
    CampaignSubmissionValue,
    CampaignSubmissionImage,
    CampaignPayoutRequest,
    CampaignPayoutLine,
    Announcement,
    MarketStudyPost,
    MarketStudySubscription,
    HourlyCryptoPrediction,
    Game, GameSession, Participation, Winner, InvalidBet
)


# ============================================================
# TokenSnapshot
# ============================================================
@admin.register(TokenSnapshot)
class TokenSnapshotAdmin(admin.ModelAdmin):
    list_display = (
        "base_symbol",
        "category",
        "market_cap",
        "liquidity_usd",
        "volume_24h",
        "txns_24h",
        "updated_at",
    )
    list_filter = ("category",)
    search_fields = ("base_symbol", "token_address", "pair_address", "dex_id")
    ordering = ("-updated_at",)


# ============================================================
# SponsoredProject
# ============================================================
@admin.register(SponsoredProject)
class SponsoredProjectAdmin(admin.ModelAdmin):
    list_display = (
        "preview_image",
        "name",
        "symbol",
        "tier",
        "is_active",
        "is_expired",
        "expires_at",
        "mcap_usd",
        "volume_24h_usd",
        "txns_24h",
        "order",
        "graduated",
        "preview_dex_icon",
    )
    list_display_links = ("name",)
    list_editable = ("order", "is_active", "graduated")
    list_filter = ("tier", "is_active", "graduated")
    search_fields = ("name", "symbol", "token_address", "source_wallet")
    ordering = ("order", "id")

    readonly_fields = (
        "preview_image",
        "preview_dex_icon",
        "dex_updated_at",
        "mcap_usd",
        "volume_24h_usd",
        "txns_24h",
        "dex_url",
        "created_at",
        "source_wallet",
    )

    fieldsets = (
        ("Basic", {
            "fields": (
                "name", "symbol", "token_address",
                "website", "description",
                "image", "preview_image",
            )
        }),
        ("Visibility / Tier", {
            "fields": (
                "tier",
                "is_active",
                "expires_at",
                "order",
                "source_wallet",
                "created_at",
            )
        }),
        ("Social", {
            "fields": (
                "x_url",
                "tg_url",
                "linkhash_holdings_pct",
                "graduated",
            )
        }),
        ("DEX (synced)", {
            "fields": (
                "dex_icon", "preview_dex_icon",
                "dex_url",
                "mcap_usd", "volume_24h_usd", "txns_24h",
                "dex_updated_at",
            )
        }),
    )

    def preview_image(self, obj):
        if obj.image and hasattr(obj.image, "url"):
            return format_html(
                '<img src="{}" style="width:56px;height:56px;border-radius:14px;object-fit:cover;" />',
                obj.image.url,
            )
        return "—"
    preview_image.short_description = "Image"

    def preview_dex_icon(self, obj):
        if obj.dex_icon and hasattr(obj.dex_icon, "url"):
            return format_html(
                '<img src="{}" style="width:28px;height:28px;border-radius:8px;object-fit:cover;background:rgba(255,255,255,0.06);padding:3px;" />',
                obj.dex_icon.url,
            )
        return "—"
    preview_dex_icon.short_description = "DEX Icon"

    def is_expired(self, obj):
        if obj.expires_at is None:
            return False
        return obj.expires_at <= timezone.now()
    is_expired.boolean = True
    is_expired.short_description = "Expired?"


# ============================================================
# UserProfile
# ============================================================
@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("wallet_address", "display_name", "points", "last_login_at", "last_seen_ip", "created_at")
    search_fields = ("wallet_address", "display_name")
    list_filter = ("created_at",)
    ordering = ("-last_login_at",)


# ============================================================
# SponsoredAdSubmission
# ============================================================
@admin.register(SponsoredAdSubmission)
class SponsoredAdSubmissionAdmin(admin.ModelAdmin):
    list_display = (
        "created_at",
        "status",
        "tier",
        "project_name",
        "submitter_wallet",
        "amount_lhx",
        "recipient_wallet",
        "tx_signature",
        "banner_preview",
    )
    list_filter = ("status", "tier", "created_at")
    search_fields = ("project_name", "tx_signature", "submitter_wallet", "token_address", "recipient_wallet")
    ordering = ("-created_at",)

    readonly_fields = (
        "created_at",
        "verified_at",
        "banner_preview",
    )

    fieldsets = (
        ("Status", {
            "fields": (
                "status",
                "note",
                "verified_at",
                "created_at",
            )
        }),
        ("Submitter / Payment", {
            "fields": (
                "submitter_wallet",
                "recipient_wallet",
                "tier",
                "amount_lhx",
                "tx_signature",
            )
        }),
        ("Project", {
            "fields": (
                "project_name",
                "symbol",
                "token_address",
                "website",
                "dex_url",
                "x_url",
                "tg_url",
                "description",
                "expires_at",
            )
        }),
        ("Sponsored Banner", {
            "fields": (
                "banner_image",
                "banner_preview",
            )
        }),
    )

    def banner_preview(self, obj):
        if obj.banner_image and hasattr(obj.banner_image, "url"):
            return format_html(
                '<a href="{}" target="_blank" rel="noopener">'
                '<img src="{}" style="height:40px;border-radius:8px;border:1px solid rgba(0,0,0,0.15);" />'
                "</a>",
                obj.banner_image.url,
                obj.banner_image.url,
            )
        return "-"
    banner_preview.short_description = "Banner"


# ============================================================
# Campaign admin (inline: fields & submissions)
# ============================================================
class CampaignFieldInline(admin.TabularInline):
    model = CampaignField
    extra = 0
    fields = ("order", "key", "label", "field_type", "required", "help_text")
    ordering = ("order", "id")


class CampaignSubmissionInline(admin.TabularInline):
    model = CampaignSubmission
    extra = 0
    can_delete = False
    ordering = ("-created_at",)

    fields = (
        "submitter_wallet",
        "status",
        "reviewer_note",
        "reviewed_at",
        "fee_paid_lhx",
        "fee_tx_signature",
        "created_at",
    )
    readonly_fields = ("created_at", "reviewed_at")


@admin.register(Campaign)
class CampaignAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "title",
        "campaign_type",
        "status",
        "created_by_wallet",
        "is_admin_created",
        "pool_total_sol",
        "submission_fee_lhx",
        "created_at",
    )
    list_filter = ("status", "campaign_type", "is_admin_created")
    search_fields = ("title", "created_by_wallet")
    readonly_fields = ("created_at", "create_verified_at")
    ordering = ("-created_at",)

    inlines = [CampaignFieldInline, CampaignSubmissionInline]


# ============================================================
# Submission detail admins
# ============================================================
class CampaignSubmissionValueInline(admin.TabularInline):
    model = CampaignSubmissionValue
    extra = 0
    can_delete = False
    fields = ("field", "value_text")


class CampaignSubmissionImageInline(admin.TabularInline):
    model = CampaignSubmissionImage
    extra = 0
    fields = ("image", "created_at")
    readonly_fields = ("created_at",)
    can_delete = True


@admin.register(CampaignSubmission)
class CampaignSubmissionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "campaign",
        "submitter_wallet",
        "status",
        "reviewer_note_short",
        "reviewed_at",
        "fee_paid_lhx",
        "created_at",
    )
    list_select_related = ("campaign",)

    # ✅ 캠페인 드롭다운 필터 제거 (캠페인이 많아지면 이게 지옥됨)
    # 대신 status/created_at 정도만 유지
    list_filter = ("status", "created_at")

    # ✅ 캠페인 선택은 autocomplete로!
    autocomplete_fields = ("campaign",)

    # ✅ 캠페인으로 분리하고 싶으면 여기서 검색하면 됨:
    # - 캠페인 제목 검색: puzzle
    # - 캠페인 ID로 검색: campaign:123
    search_fields = (
        "submitter_wallet",
        "fee_tx_signature",
        "campaign__title",
        "reviewer_note",
    )
    ordering = ("-created_at",)
    readonly_fields = ("created_at", "fee_verified_at", "reviewed_at")

    inlines = [CampaignSubmissionValueInline, CampaignSubmissionImageInline]

    def reviewer_note_short(self, obj):
        txt = (obj.reviewer_note or "").strip()
        if not txt:
            return ""
        return txt[:40] + ("…" if len(txt) > 40 else "")
    reviewer_note_short.short_description = "Note"

    # ✅ (선택) 검색창에 "campaign:2" 입력하면 campaign_id=2로 필터되게
    def get_search_results(self, request, queryset, search_term):
        qs, use_distinct = super().get_search_results(request, queryset, search_term)
        term = (search_term or "").strip()
        if term.startswith("campaign:"):
            raw = term.split("campaign:", 1)[1].strip()
            if raw.isdigit():
                qs = qs.filter(campaign_id=int(raw))
        return qs, use_distinct



@admin.register(CampaignPayoutRequest)
class CampaignPayoutRequestAdmin(admin.ModelAdmin):
    list_display = ("id", "campaign", "status", "requested_by_wallet", "total_recipients", "total_amount_sol", "created_at")
    list_filter = ("status", "created_at")
    search_fields = ("campaign__title", "requested_by_wallet")
    actions = ["mark_processing", "mark_paid", "mark_rejected"]

    @admin.action(description="Mark selected requests as PROCESSING")
    def mark_processing(self, request, queryset):
        queryset.update(status=CampaignPayoutRequest.STATUS_PROCESSING)

    @admin.action(description="Mark selected requests as PAID (and mark submissions as PAID)")
    def mark_paid(self, request, queryset):
        now = timezone.now()
        with transaction.atomic():
            for req in queryset.select_for_update():
                req.status = CampaignPayoutRequest.STATUS_PAID
                req.processed_at = now
                req.save(update_fields=["status", "processed_at"])

                # ✅ 해당 request에 묶인 submissions를 paid로
                CampaignSubmission.objects.filter(
                    payout_request=req,
                    payout_status=CampaignSubmission.PAYOUT_RESERVED,
                ).update(
                    payout_status=CampaignSubmission.PAYOUT_PAID,
                    paid_at=now,
                )

    @admin.action(description="Mark selected requests as REJECTED (and release submissions to NONE)")
    def mark_rejected(self, request, queryset):
        with transaction.atomic():
            for req in queryset.select_for_update():
                req.status = CampaignPayoutRequest.STATUS_REJECTED
                req.processed_at = timezone.now()
                req.save(update_fields=["status", "processed_at"])

                # ✅ reserved 해제
                CampaignSubmission.objects.filter(
                    payout_request=req,
                    payout_status=CampaignSubmission.PAYOUT_RESERVED,
                ).update(
                    payout_status=CampaignSubmission.PAYOUT_NONE,
                    payout_request=None,
                    payout_reserved_at=None,
                )

                # ✅ (추가) payout lines 삭제 -> 재요청 가능
                CampaignPayoutLine.objects.filter(request=req).delete()


@admin.register(CampaignPayoutLine)
class CampaignPayoutLineAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "request",
        "recipient_wallet",
        "amount_sol",
        "payout_status",
        "sent_at",
        "payout_tx_signature",
        "submission",
        "created_at",
    )
    list_filter = ("payout_status", "request__campaign", "created_at")
    search_fields = ("recipient_wallet", "payout_tx_signature", "request__campaign__title")
    ordering = ("-created_at",)

    list_editable = ("payout_status", "payout_tx_signature")  # ✅ 리스트에서 바로 변경 가능 (원하면 제거)
    readonly_fields = ("created_at",)

    actions = ["mark_as_sent", "mark_as_pending", "export_as_csv"]

    @admin.action(description="Mark selected lines as SENT (set sent_at=now if empty)")
    def mark_as_sent(self, request, queryset):
        now = timezone.now()
        with transaction.atomic():
            qs = queryset.select_for_update()
            qs.filter(sent_at__isnull=True).update(
                payout_status=CampaignPayoutLine.STATUS_SENT,
                sent_at=now,
            )
            qs.filter(sent_at__isnull=False).update(
                payout_status=CampaignPayoutLine.STATUS_SENT,
            )

    @admin.action(description="Mark selected lines as PENDING (clear sent_at / tx)")
    def mark_as_pending(self, request, queryset):
        with transaction.atomic():
            queryset.select_for_update().update(
                payout_status=CampaignPayoutLine.STATUS_PENDING,
                sent_at=None,
                payout_tx_signature="",
            )

    @admin.action(description="Export selected payout lines to CSV")
    def export_as_csv(self, request, queryset):
        # CSV 응답
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="campaign_payout_lines.csv"'
        writer = csv.writer(response)

        # 헤더
        writer.writerow([
            "line_id",
            "campaign_id",
            "campaign_title",
            "payout_request_id",
            "recipient_wallet",
            "amount_sol",
            "line_status",
            "sent_at",
            "tx_signature",
            "submission_id",
            "submission_status",
            "submission_wallet",
            "created_at",
        ])

        # 필요한 연관 미리 로딩
        qs = queryset.select_related("request", "request__campaign", "submission")

        for line in qs:
            campaign = line.request.campaign if line.request_id else None
            sub = line.submission

            writer.writerow([
                line.id,
                campaign.id if campaign else "",
                campaign.title if campaign else "",
                line.request_id or "",
                line.recipient_wallet,
                str(line.amount_sol),
                line.payout_status,
                line.sent_at.isoformat() if line.sent_at else "",
                line.payout_tx_signature,
                sub.id if sub else "",
                sub.status if sub else "",
                sub.submitter_wallet if sub else "",
                line.created_at.isoformat() if line.created_at else "",
            ])

        return response


@admin.register(Announcement)
class AnnouncementAdmin(admin.ModelAdmin):
    list_display = ("title", "is_published", "pinned", "published_at", "created_at")
    list_filter = ("is_published", "pinned")
    search_fields = ("title", "content")
    prepopulated_fields = {"slug": ("title",)}
    ordering = ("-pinned", "-published_at", "-created_at")
    date_hierarchy = "created_at"

    fieldsets = (
        (None, {"fields": ("title", "slug")}),
        ("Content", {"fields": ("content",)}),
        ("Publishing", {"fields": ("is_published", "pinned", "published_at")}),
    )

@admin.register(MarketStudyPost)
class MarketStudyPostAdmin(admin.ModelAdmin):
    """
    MarketStudyPost.content 를 CKEditor(이미지 업로더 포함)로 편집 가능하게.
    """

    class Form(forms.ModelForm):
        content = forms.CharField(
            required=False,
            widget=CKEditorUploadingWidget(),
            help_text="You can embed images, links, and rich formatting.",
        )

        class Meta:
            model = MarketStudyPost
            fields = "__all__"

    form = Form

    list_display = (
        "title",
        "slug",
        "is_free",
        "is_published",
        "created_at",
    )
    list_filter = ("is_free", "is_published", "created_at")
    search_fields = ("title", "slug", "excerpt", "content")
    prepopulated_fields = {"slug": ("title",)}
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)

    fieldsets = (
        (None, {"fields": ("title", "slug")}),
        ("Summary", {"fields": ("excerpt",)}),
        ("Content", {"fields": ("content",)}),  # ✅ CKEditor here
        ("Publishing", {"fields": ("is_free", "is_published")}),
        ("Meta", {"fields": ("created_at",)}),
    )


# ============================================================
# Market Study Subscriptions
# ============================================================
@admin.register(MarketStudySubscription)
class MarketStudySubscriptionAdmin(admin.ModelAdmin):
    """
    Admin에서 구독 상태/결제/만료를 한 눈에 보이게
    """
    list_display = (
        "wallet_address",
        "asset",
        "amount",
        "status",
        "verified_at",
        "expires_at",
        "is_active_now",
        "tx_signature_short",
        "created_at",
    )
    list_filter = (
        "asset",
        "status",
        "created_at",
        "expires_at",
    )
    search_fields = (
        "wallet_address",
        "tx_signature",
        "note",
    )
    ordering = ("-created_at",)

    readonly_fields = (
        "created_at",
    )

    fieldsets = (
        ("Subscriber", {
            "fields": ("wallet_address",),
        }),
        ("Payment", {
            "fields": ("asset", "amount", "tx_signature"),
        }),
        ("Status", {
            "fields": ("status", "verified_at", "expires_at"),
        }),
        ("Internal", {
            "fields": ("note", "created_at"),
        }),
    )

    @admin.display(boolean=True, description="Active")
    def is_active_now(self, obj: MarketStudySubscription):
        try:
            return obj.is_active()
        except Exception:
            return False

    @admin.display(description="Tx (short)")
    def tx_signature_short(self, obj: MarketStudySubscription):
        s = (obj.tx_signature or "").strip()
        if not s:
            return "-"
        if len(s) <= 14:
            return s
        return f"{s[:6]}...{s[-6:]}"
    

@admin.register(HourlyCryptoPrediction)
class HourlyCryptoPredictionAdmin(admin.ModelAdmin):
    list_display = ("asset", "verdict", "as_of_utc", "next_update_utc", "updated_at")
    list_filter = ("asset", "verdict")
    search_fields = ("asset",)
    readonly_fields = ("created_at", "updated_at")
    ordering = ("asset",)

# ============================================================
# ✅ GAMES ADMIN (NEW)
# ============================================================

# --- Inlines ---
class GameSessionInline(admin.TabularInline):
    model = GameSession
    extra = 0
    fields = (
        "round_no",
        "status",
        "pot_total_sol",
        "fee_pct_snapshot",
        "fomo_ends_at",
        "lottery_draw_at",
        "winning_numbers",
        "created_at",
    )
    readonly_fields = ("created_at",)
    ordering = ("-created_at",)

class InvalidBetInline(admin.TabularInline):
    model = InvalidBet
    extra = 0
    fields = (
        "created_at",
        "refund_status",
        "reason",
        "submitter_wallet",
        "tx_signature",
        "amount_onchain_sol",
        "refund_tx_signature",
        "refunded_at",
    )
    readonly_fields = ("created_at",)
    ordering = ("-created_at",)
    show_change_link = True

class ParticipationInline(admin.TabularInline):
    model = Participation
    extra = 0
    fields = (
        "created_at",
        "user_profile",
        "kind",
        "status",
        "amount_sol",
        "tx_signature",
        "picked_numbers",
        "bet_index",
        "verified_at",
        "reject_reason",
    )
    readonly_fields = ("created_at",)
    ordering = ("-created_at",)
    show_change_link = True


class WinnerInline(admin.TabularInline):
    model = Winner
    extra = 0
    fields = (
        "created_at",
        "user_profile",
        "participation",
        "payout_amount_sol",
        "payout_status",
        "payout_tx_signature",
        "paid_at",
    )
    readonly_fields = ("created_at",)
    ordering = ("-created_at",)
    show_change_link = True


# --- Game admin ---
@admin.register(Game)
class GameAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "name",
        "category",
        "subtype",
        "protocol_fee_pct",
        "min_bet_sol",
        "fixed_bet_sol",
        "increment_pct",
        "fomo_timer_seconds",
        "lottery_pick_count",
        "lottery_number_max",
        "lottery_rollover_enabled",
        "is_active",
        "created_at",
    )
    list_filter = ("category", "subtype", "is_active", "lottery_rollover_enabled")
    search_fields = ("name", "sol_public_key")
    list_editable = ("is_active",)
    readonly_fields = ("created_at",)

    fieldsets = (
        ("Core", {
            "fields": (
                "name",
                "category",
                "subtype",
                "description",
                "is_active",
                "protocol_fee_pct",
                "created_at",
            )
        }),
        ("Solana Wallet (Game Wallet)", {
            "fields": (
                "sol_public_key",
                "sol_private_key",
            )
        }),
        ("FOMO Rules", {
            "fields": (
                "fomo_timer_seconds",
                "min_bet_sol",
                "fixed_bet_sol",
                "increment_pct",
            )
        }),
        ("Lottery Rules", {
            "fields": (
                "lottery_pick_count",
                "lottery_number_max",
                "lottery_rollover_enabled",
            )
        }),
    )

    inlines = []


# --- GameSession admin ---
@admin.register(GameSession)
class GameSessionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "game",
        "round_no",
        "status",
        "pot_total_sol",
        "fee_pct_snapshot",
        "fomo_ends_at",
        "lottery_draw_at",
        "winning_numbers",
        "settled_at",
        "created_at",
    )
    list_filter = ("status", "game__category", "game__subtype", "game")
    search_fields = ("game__name", "winning_numbers", "settlement_tx_signature")
    list_editable = ("status",)
    readonly_fields = ("created_at",)

    fieldsets = (
        ("Core", {
            "fields": (
                "game",
                "round_no",
                "status",
                "created_at",
            )
        }),
        ("Pot / Fee", {
            "fields": (
                "pot_total_sol",
                "fee_pct_snapshot",
            )
        }),
        ("FOMO", {
            "fields": (
                "fomo_ends_at",
            )
        }),
        ("Lottery", {
            "fields": (
                "lottery_draw_at",
                "winning_numbers",
                "rollover_from",
            )
        }),
        ("Settlement", {
            "fields": (
                "settled_at",
                "settlement_tx_signature",
            )
        }),
    )

    inlines = [ParticipationInline, WinnerInline, InvalidBetInline]

# --- Participation admin ---
@admin.register(Participation)
class ParticipationAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "session",
        "game_name",
        "user_profile",
        "kind",
        "status",
        "amount_sol",
        "bet_index",
        "picked_numbers",
        "tx_signature",
        "verified_at",
        "created_at",
    )
    list_filter = ("kind", "status", "session__game__category", "session__game__subtype", "session__game")
    search_fields = ("tx_signature", "user_profile__wallet_address", "session__game__name")
    list_editable = ("status",)
    readonly_fields = ("created_at",)

    fieldsets = (
        ("Core", {
            "fields": (
                "session",
                "user_profile",
                "kind",
                "status",
                "created_at",
            )
        }),
        ("Payment Proof", {
            "fields": (
                "amount_sol",
                "tx_signature",
                "verified_at",
            )
        }),
        ("Game Data", {
            "fields": (
                "picked_numbers",
                "bet_index",
            )
        }),
        ("Admin Notes", {
            "fields": (
                "reject_reason",
            )
        }),
    )

    def game_name(self, obj):
        try:
            return obj.session.game.name
        except Exception:
            return "-"
    game_name.short_description = "Game"


# --- Winner admin ---
@admin.register(Winner)
class WinnerAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "session",
        "game_name",
        "user_profile",
        "payout_amount_sol",
        "payout_status",
        "payout_tx_signature",
        "paid_at",
        "created_at",
    )
    list_filter = ("payout_status", "session__game__category", "session__game__subtype", "session__game")
    search_fields = ("user_profile__wallet_address", "session__game__name", "payout_tx_signature")
    list_editable = ("payout_status",)
    readonly_fields = ("created_at",)

    fieldsets = (
        ("Core", {
            "fields": (
                "session",
                "user_profile",
                "participation",
                "created_at",
            )
        }),
        ("Payout", {
            "fields": (
                "payout_amount_sol",
                "payout_status",
                "payout_tx_signature",
                "paid_at",
            )
        }),
    )

    def game_name(self, obj):
        try:
            return obj.session.game.name
        except Exception:
            return "-"
    game_name.short_description = "Game"


@admin.register(InvalidBet)
class InvalidBetAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "created_at",
        "refund_status",
        "reason",
        "game",
        "session_id_display",
        "submitter_wallet_short",
        "amount_onchain_sol",
        "tx_signature_short",
        "refund_tx_signature_short",
    )
    list_filter = (
        "refund_status",
        "reason",
        "game",
        "session__status",
        "game__subtype",
        "created_at",
    )
    search_fields = (
        "tx_signature",
        "refund_tx_signature",
        "submitter_wallet",
        "receiver_wallet",
        "game__name",
        "session__id",
    )
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)

    fieldsets = (
        ("Core", {
            "fields": (
                "created_at",
                "refund_status",
                "reason",
                "game",
                "session",
            )
        }),
        ("Transaction", {
            "fields": (
                "submitter_wallet",
                "receiver_wallet",
                "tx_signature",
                "amount_onchain_sol",
                "block_time",
                "note",
            )
        }),
        ("Refund", {
            "fields": (
                "refund_tx_signature",
                "refunded_at",
                "admin_note",
            )
        }),
    )

    actions = ["mark_refunded", "mark_pending", "mark_ignored"]

    def session_id_display(self, obj):
        return obj.session_id or "-"
    session_id_display.short_description = "Session ID"

    def tx_signature_short(self, obj):
        s = (obj.tx_signature or "").strip()
        if not s:
            return "-"
        return f"{s[:8]}...{s[-6:]}" if len(s) > 18 else s
    tx_signature_short.short_description = "TX"

    def refund_tx_signature_short(self, obj):
        s = (obj.refund_tx_signature or "").strip()
        if not s:
            return "-"
        return f"{s[:8]}...{s[-6:]}" if len(s) > 18 else s
    refund_tx_signature_short.short_description = "Refund TX"

    def submitter_wallet_short(self, obj):
        w = (obj.submitter_wallet or "").strip()
        if not w:
            return "-"
        return f"{w[:4]}...{w[-4:]}" if len(w) > 12 else w
    submitter_wallet_short.short_description = "Submitter"

    def mark_refunded(self, request, queryset):
        now = timezone.now()
        queryset.update(refund_status=InvalidBet.REFUND_REFUNDED, refunded_at=now)
    mark_refunded.short_description = "Mark selected as REFUNDED (sets refunded_at=now)"

    def mark_pending(self, request, queryset):
        queryset.update(refund_status=InvalidBet.REFUND_PENDING)
    mark_pending.short_description = "Mark selected as PENDING"

    def mark_ignored(self, request, queryset):
        queryset.update(refund_status=InvalidBet.REFUND_IGNORED)
    mark_ignored.short_description = "Mark selected as IGNORED"
