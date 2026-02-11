from django.db import models
from django.utils import timezone
from datetime import timedelta
from decimal import Decimal
from django.core.validators import MinValueValidator
from datetime import timezone as dt_timezone
from ckeditor_uploader.fields import RichTextUploadingField
from django.utils.text import slugify
from django.utils import timezone

# ============================================================
# TokenSnapshot
# ============================================================
class TokenSnapshot(models.Model):
    """
    DexScreener 기반 토큰 스냅샷
    (주기적 업데이트 / 읽기 전용 성격)
    """
    token_address = models.CharField(max_length=64, db_index=True)
    pair_address = models.CharField(max_length=64, blank=True, null=True)
    dex_id = models.CharField(max_length=32, blank=True, null=True)

    base_symbol = models.CharField(max_length=32, blank=True, null=True)

    price_usd = models.FloatField(blank=True, null=True)
    liquidity_usd = models.FloatField(blank=True, null=True)
    volume_24h = models.FloatField(blank=True, null=True)
    txns_24h = models.IntegerField(blank=True, null=True)

    fdv = models.FloatField(blank=True, null=True)
    market_cap = models.FloatField(blank=True, null=True)

    url = models.URLField(blank=True, null=True)

    vol_mcap_ratio = models.FloatField(blank=True, null=True)
    lp_mcap_ratio = models.FloatField(blank=True, null=True)
    potential_score_hits = models.PositiveSmallIntegerField(default=0)

    CATEGORY_CHOICES = [
        ("major", "Major"),
        ("potential", "Potential"),
    ]
    category = models.CharField(
        max_length=16,
        choices=CATEGORY_CHOICES,
        db_index=True,
    )

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["category", "-market_cap"]),
            models.Index(fields=["category", "-volume_24h"]),
            models.Index(fields=["category", "-potential_score_hits"]),
        ]

    def __str__(self):
        return f"{self.base_symbol or 'UNK'} ({self.category})"


# ============================================================
# SponsoredProject
# ============================================================
class SponsoredProject(models.Model):
    """
    LinkHash가 광고/프로모션하는 프로젝트 (Admin에서 수동 관리)
    + 일부 지표는 DexScreener에서 가져와 저장해둠(동기화)
    """

    # =========================
    # NEW: Chain / Launchpad
    # =========================
    CHAIN_SOL = "sol"
    CHAIN_BNB = "bnb"
    CHAIN_OTHERS = "others"
    CHAIN_CHOICES = [
        (CHAIN_SOL, "Solana"),
        (CHAIN_BNB, "BNB Chain"),
        (CHAIN_OTHERS, "Others"),
    ]

    LAUNCHPAD_NONE = "none"
    LAUNCHPAD_PUMPFUN = "pumpfun"
    LAUNCHPAD_FOURMEME = "fourmeme"
    LAUNCHPAD_CHOICES = [
        (LAUNCHPAD_NONE, "None"),
        (LAUNCHPAD_PUMPFUN, "pump.fun"),
        (LAUNCHPAD_FOURMEME, "four.meme"),
    ]

    # --- 기본 정보 (기존 유지) ---
    name = models.CharField(max_length=64)
    symbol = models.CharField(max_length=32, blank=True, null=True)
    token_address = models.CharField(max_length=64, blank=True, null=True)

    website = models.URLField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)

    image = models.ImageField(
        upload_to="sponsored/",
        blank=True,
        null=True,
        help_text="권장 비율: 1:1 또는 16:9"
    )

    order = models.PositiveIntegerField(default=0, db_index=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # --- 추가: X / Telegram ---
    x_url = models.URLField(blank=True, null=True, help_text="X(Twitter) link")
    tg_url = models.URLField(blank=True, null=True, help_text="Telegram link")

    # --- 추가: LinkHash Holdings (%)
    # 예: 1.25% -> 1.25 저장
    linkhash_holdings_pct = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        blank=True,
        null=True,
        help_text="LinkHash holdings percentage (e.g., 1.25)"
    )

    # ✅ NEW: 체인 (SOL/BNB)
    chain = models.CharField(
        max_length=8,
        choices=CHAIN_CHOICES,
        default=CHAIN_SOL,
        db_index=True,
        help_text="Underlying chain for the token (Solana / BNB Chain)."
    )

    # ✅ NEW: 런치패드 출신 여부 (optional)
    launchpad = models.CharField(
        max_length=16,
        choices=LAUNCHPAD_CHOICES,
        default=LAUNCHPAD_NONE,
        blank=True,
        help_text="If the token originates from a launchpad (pump.fun / four.meme)."
    )

    # --- 졸업(Graduated) 여부 ---
    # launchpad가 none이면 의미가 없지만, 일단 UI/필터/뱃지용으로 유지
    graduated = models.BooleanField(default=False)

    # --- 추가: DEX 아이콘 업로드(버튼 옆에 표시용) ---
    dex_icon = models.ImageField(
        upload_to="sponsored/dex_icons/",
        blank=True,
        null=True,
        help_text="DEX icon (e.g., Raydium/Jupiter icon)"
    )

    # --- DexScreener에서 계산/수집해서 저장할 필드들 ---
    # (Admin에서 직접 입력하지 않고, sync로 갱신하는 용도 권장)
    mcap_usd = models.FloatField(blank=True, null=True)
    volume_24h_usd = models.FloatField(blank=True, null=True)
    txns_24h = models.IntegerField(blank=True, null=True)
    dex_url = models.URLField(blank=True, null=True, help_text="DexScreener or DEX link (auto)")

    dex_updated_at = models.DateTimeField(blank=True, null=True)

    # tier / expires / source
    tier = models.CharField(max_length=16, default="basic")  # basic/premium/best_partner 등 확장 가능
    expires_at = models.DateTimeField(null=True, blank=True)
    source_wallet = models.CharField(max_length=64, blank=True, default="")

    # cex, dex links
    # SponsoredProject 모델 안에 추가
    cex_links = models.JSONField(blank=True, default=list, help_text="Optional: up to 10 CEX URLs")
    dex_links = models.JSONField(blank=True, default=list,
                                 help_text="Optional: up to 3 DEX/launchpad URLs (pump.fun etc.)")

    class Meta:
        ordering = ["order", "id"]
        indexes = [
            models.Index(fields=["is_active", "order"]),
            models.Index(fields=["graduated", "order"]),
            models.Index(fields=["chain", "order"]),
            models.Index(fields=["launchpad", "order"]),
        ]

    def __str__(self):
        return self.name


from django.db import models

class UserProfile(models.Model):
    wallet_address = models.CharField(
        max_length=64,
        unique=True,
        db_index=True
    )

    display_name = models.CharField(
        max_length=32,
        blank=True,
        default=""
    )

    bio = models.TextField(
        blank=True,
        default=""
    )

    # ✅ 추가: 아바타 이미지
    avatar = models.ImageField(
        upload_to="avatars/",
        blank=True,
        null=True
    )

    created_at = models.DateTimeField(auto_now_add=True)
    last_login_at = models.DateTimeField(null=True, blank=True)
    last_seen_ip = models.GenericIPAddressField(null=True, blank=True)

    points = models.IntegerField(default=0)

    def __str__(self):
        return self.wallet_address


class AdSubmission(models.Model):
    TIER_BASIC = "basic"
    TIER_PREMIUM = "premium"
    TIER_CHOICES = [(TIER_BASIC, "Basic"), (TIER_PREMIUM, "Premium")]

    STATUS_PENDING = "pending"
    STATUS_VERIFIED = "verified"
    STATUS_REJECTED = "rejected"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_VERIFIED, "Verified"),
        (STATUS_REJECTED, "Rejected"),
    ]

    wallet_address = models.CharField(max_length=64, db_index=True)
    tier = models.CharField(max_length=16, choices=TIER_CHOICES)
    tx_signature = models.CharField(max_length=128, unique=True, db_index=True)

    project_name = models.CharField(max_length=80)
    symbol = models.CharField(max_length=20, blank=True, default="")
    token_address = models.CharField(max_length=64, blank=True, default="")
    website = models.URLField(blank=True, default="")
    x_url = models.URLField(blank=True, default="")
    tg_url = models.URLField(blank=True, default="")
    dex_url = models.URLField(blank=True, default="")
    description = models.TextField(blank=True, default="")

    amount_lhx = models.BigIntegerField(default=0)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING)
    verified_at = models.DateTimeField(null=True, blank=True)
    reject_reason = models.CharField(max_length=255, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.tier} {self.project_name} ({self.status})"


class SponsoredAdSubmission(models.Model):
    TIER_BASIC = "basic"
    TIER_PREMIUM = "premium"
    TIER_CHOICES = [
        (TIER_BASIC, "Basic"),
        (TIER_PREMIUM, "Premium"),
    ]

    STATUS_PENDING = "pending"
    STATUS_VERIFIED = "verified"
    STATUS_REJECTED = "rejected"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_VERIFIED, "Verified"),
        (STATUS_REJECTED, "Rejected"),
    ]

    # =========================
    # NEW: Chain / Launchpad
    # =========================
    CHAIN_SOL = "sol"
    CHAIN_BNB = "bnb"
    CHAIN_OTHERS = "others"
    CHAIN_CHOICES = [
        (CHAIN_SOL, "Solana"),
        (CHAIN_BNB, "BNB Chain"),
        (CHAIN_OTHERS, "Others"),
    ]

    LAUNCHPAD_NONE = "none"
    LAUNCHPAD_PUMPFUN = "pumpfun"
    LAUNCHPAD_FOURMEME = "fourmeme"
    LAUNCHPAD_CHOICES = [
        (LAUNCHPAD_NONE, "None"),
        (LAUNCHPAD_PUMPFUN, "pump.fun"),
        (LAUNCHPAD_FOURMEME, "four.meme"),
    ]

    # who
    submitter_wallet = models.CharField(max_length=64, db_index=True)

    # payment
    tier = models.CharField(max_length=16, choices=TIER_CHOICES)
    tx_signature = models.CharField(max_length=128, unique=True, db_index=True)
    amount_lhx = models.BigIntegerField(default=0)  # “토큰 단위(정수)”로 저장: 100000 LHX 그대로
    recipient_wallet = models.CharField(max_length=64)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING)

    # project info
    project_name = models.CharField(max_length=64)
    symbol = models.CharField(max_length=32, blank=True, default="")
    token_address = models.CharField(max_length=64, blank=True, default="")
    website = models.URLField(blank=True, default="")
    dex_url = models.URLField(blank=True, default="")
    x_url = models.URLField(blank=True, default="")
    tg_url = models.URLField(blank=True, default="")
    description = models.TextField(blank=True, default="")

    # SponsoredAdSubmission 모델 안에 추가
    cex_links = models.JSONField(blank=True, default=list, help_text="Optional: up to 10 CEX URLs")
    dex_links = models.JSONField(blank=True, default=list,
                                 help_text="Optional: up to 3 DEX/launchpad URLs (pump.fun etc.)")

    # ✅ NEW: chain 선택 (SOL/BNB)
    chain = models.CharField(
        max_length=8,
        choices=CHAIN_CHOICES,
        default=CHAIN_SOL,
        db_index=True,
        help_text="Underlying chain for the token (Solana / BNB Chain)."
    )

    # ✅ NEW: launchpad 출신 (optional)
    launchpad = models.CharField(
        max_length=16,
        choices=LAUNCHPAD_CHOICES,
        default=LAUNCHPAD_NONE,
        blank=True,
        help_text="If the token originates from a launchpad (pump.fun / four.meme)."
    )

    # ✅ NEW: graduated (optional)
    graduated = models.BooleanField(
        default=False,
        help_text="Whether the token has graduated from the launchpad (if applicable)."
    )

    # admin/debug
    verified_at = models.DateTimeField(null=True, blank=True)
    starts_at = models.DateTimeField(null=True, blank=True, db_index=True)
    expires_at = models.DateTimeField(null=True, blank=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    note = models.TextField(blank=True, default="")  # 검증 실패 사유 등

    banner_image = models.ImageField(
        upload_to="sponsored_ads/",
        null=True,
        blank=True
    )

    def __str__(self):
        return f"{self.project_name} ({self.tier}) - {self.status}"

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["chain", "-created_at"]),
            models.Index(fields=["launchpad", "-created_at"]),
            models.Index(fields=["graduated", "-created_at"]),
        ]



class Campaign(models.Model):
    TYPE_DEX_PROMO = "dex_promo"
    TYPE_GENERAL = "general"
    TYPE_CHOICES = [
        (TYPE_DEX_PROMO, "DEX Project Promo"),
        (TYPE_GENERAL, "General"),
    ]

    STATUS_DRAFT = "draft"
    STATUS_ACTIVE = "active"
    STATUS_ENDED = "ended"   # ✅ past campaign 역할
    STATUS_CHOICES = [
        (STATUS_DRAFT, "Draft"),
        (STATUS_ACTIVE, "Active"),
        (STATUS_ENDED, "Ended"),
    ]

    # ✅ 기간 옵션 (7/14/30)
    DURATION_7 = 7
    DURATION_14 = 14
    DURATION_30 = 30
    DURATION_CHOICES = [
        (DURATION_7, "7 days"),
        (DURATION_14, "14 days"),
        (DURATION_30, "30 days"),
    ]
    duration_days = models.PositiveSmallIntegerField(
        choices=DURATION_CHOICES,
        default=DURATION_7,
        help_text="How long this campaign lasts (auto sets ends_at).",
    )

    # 기본
    title = models.CharField(max_length=120)
    subtitle = models.CharField(max_length=200, blank=True, default="")
    description = models.TextField(blank=True, default="")
    cover_image = models.ImageField(upload_to="campaigns/covers/", blank=True, null=True)

    campaign_type = models.CharField(max_length=16, choices=TYPE_CHOICES, db_index=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_DRAFT, db_index=True)

    # 생성자/주체
    created_by_wallet = models.CharField(max_length=64, blank=True, default="", db_index=True)
    is_admin_created = models.BooleanField(default=False)

    # 캠페인 기간
    starts_at = models.DateTimeField(null=True, blank=True, db_index=True)
    ends_at = models.DateTimeField(null=True, blank=True, db_index=True)

    # 제출 정책
    submission_fee_lhx = models.PositiveIntegerField(default=100)  # 기본 100 LHX

    # ✅ 풀/리워드 (SOL 기준) - 최소 0.05 SOL 강제 (모델 레벨)
    pool_total_sol = models.DecimalField(
        max_digits=18,
        decimal_places=9,
        null=True,
        blank=True,
        validators=[MinValueValidator(Decimal("0.05"))],
    )
    reward_per_user_sol = models.DecimalField(max_digits=18, decimal_places=9, null=True, blank=True)
    distribute_mode = models.CharField(
        max_length=16,
        default="per_user",
        help_text="per_user or split_n",
    )
    platform_fee_pct = models.DecimalField(max_digits=5, decimal_places=2, default=5.00)  # 5%

    # 파트너 캠페인 생성 결제 검증용
    create_tx_signature = models.CharField(max_length=128, blank=True, default="", db_index=True)
    create_verified_at = models.DateTimeField(null=True, blank=True)

    # 검증 도움 안내
    needs_verification_help_text = models.BooleanField(default=True)
    verification_help_cost_sol = models.DecimalField(max_digits=18, decimal_places=9, default=2)
    verification_help_contact = models.CharField(max_length=64, default="@lhxluke")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status", "-created_at"]),
            models.Index(fields=["campaign_type", "status"]),
            models.Index(fields=["ends_at", "status"]),
        ]

    def __str__(self):
        return f"{self.title} ({self.campaign_type})"

    # ✅ 생성/수정 시 ends_at 자동 세팅 도우미
    def set_default_schedule_if_missing(self):
        now = timezone.now()
        if not self.starts_at:
            self.starts_at = now
        if not self.ends_at:
            self.ends_at = self.starts_at + timedelta(days=int(self.duration_days))

    # ✅ 만료 체크
    def is_expired(self):
        return bool(self.ends_at and self.ends_at < timezone.now())


class CampaignField(models.Model):
    FIELD_TEXT = "text"
    FIELD_URL = "url"
    FIELD_WALLET = "wallet"
    FIELD_NUMBER = "number"
    FIELD_CHOICES = [
        (FIELD_TEXT, "Text"),
        (FIELD_URL, "URL"),
        (FIELD_WALLET, "Wallet"),
        (FIELD_NUMBER, "Number"),
    ]

    campaign = models.ForeignKey(Campaign, on_delete=models.CASCADE, related_name="fields")

    key = models.SlugField(max_length=40, help_text="ex) x_username, tg_username, wallet")
    label = models.CharField(max_length=80)
    help_text = models.CharField(max_length=160, blank=True, default="")
    field_type = models.CharField(max_length=16, choices=FIELD_CHOICES, default=FIELD_TEXT)

    required = models.BooleanField(default=True)
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["order", "id"]
        unique_together = [("campaign", "key")]

    def __str__(self):
        return f"{self.campaign.title} - {self.key}"


class CampaignSubmission(models.Model):
    STATUS_PENDING = "pending"
    STATUS_APPROVED = "approved"
    STATUS_REJECTED = "rejected"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_APPROVED, "Approved"),
        (STATUS_REJECTED, "Rejected"),
    ]

    # ✅ payout tracking (중복 방지 + 운영 구분)
    PAYOUT_NONE = "none"
    PAYOUT_RESERVED = "reserved"
    PAYOUT_PAID = "paid"
    PAYOUT_STATUS_CHOICES = [
        (PAYOUT_NONE, "None"),
        (PAYOUT_RESERVED, "Reserved"),
        (PAYOUT_PAID, "Paid"),
    ]

    campaign = models.ForeignKey(Campaign, on_delete=models.CASCADE, related_name="submissions")
    submitter_wallet = models.CharField(max_length=64, db_index=True)

    # 100 LHX 제출 수수료
    fee_paid_lhx = models.PositiveIntegerField(default=0)
    fee_tx_signature = models.CharField(max_length=128, blank=True, default="", db_index=True)
    fee_verified_at = models.DateTimeField(null=True, blank=True)

    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING, db_index=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)
    reviewer_note = models.CharField(max_length=255, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)

    # creator가 마킹했는지 추적
    reviewed_by_wallet = models.CharField(max_length=64, blank=True, default="", db_index=True)

    # ✅ NEW: payout 상태/연결 (중복 요청 방지 핵심)
    payout_status = models.CharField(
        max_length=16,
        choices=PAYOUT_STATUS_CHOICES,
        default=PAYOUT_NONE,
        db_index=True,
    )
    payout_request = models.ForeignKey(
        "CampaignPayoutRequest",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="reserved_submissions",
    )
    payout_reserved_at = models.DateTimeField(null=True, blank=True)
    paid_at = models.DateTimeField(null=True, blank=True)
    payout_tx = models.CharField(max_length=128, blank=True, default="")  # optional (실제 지급 tx)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["campaign", "status", "-created_at"]),
            models.Index(fields=["submitter_wallet", "-created_at"]),
            # ✅ payout 조회 최적화
            models.Index(fields=["campaign", "payout_status", "-created_at"]),
        ]

    def __str__(self):
        return f"{self.campaign.title} - {self.submitter_wallet[:6]}... ({self.status})"


class CampaignSubmissionValue(models.Model):
    submission = models.ForeignKey(CampaignSubmission, on_delete=models.CASCADE, related_name="values")
    field = models.ForeignKey(CampaignField, on_delete=models.CASCADE)

    value_text = models.TextField(blank=True, default="")

    class Meta:
        unique_together = [("submission", "field")]

    def __str__(self):
        return f"{self.field.key}={self.value_text[:20]}"


class CampaignSubmissionImage(models.Model):
    submission = models.ForeignKey(CampaignSubmission, on_delete=models.CASCADE, related_name="images")
    image = models.ImageField(upload_to="campaigns/submissions/")
    created_at = models.DateTimeField(auto_now_add=True)

class CampaignPayoutRequest(models.Model):
    STATUS_REQUESTED = "requested"
    STATUS_PROCESSING = "processing"
    STATUS_PAID = "paid"
    STATUS_REJECTED = "rejected"
    STATUS_CHOICES = [
        (STATUS_REQUESTED, "Requested"),
        (STATUS_PROCESSING, "Processing"),
        (STATUS_PAID, "Paid"),
        (STATUS_REJECTED, "Rejected"),
    ]

    campaign = models.ForeignKey(Campaign, on_delete=models.CASCADE, related_name="payout_requests")
    requested_by_wallet = models.CharField(max_length=64, db_index=True)

    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_REQUESTED, db_index=True)
    memo = models.CharField(max_length=255, blank=True, default="")

    # 스냅샷 합계(관리 편의)
    total_recipients = models.PositiveIntegerField(default=0)
    total_amount_sol = models.DecimalField(max_digits=18, decimal_places=9, default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"PayoutRequest #{self.id} - {self.campaign.title}"


class CampaignPayoutLine(models.Model):
    STATUS_PENDING = "pending"  # 아직 안 보냄
    STATUS_SENT = "sent"  # 보냄
    STATUS_FAILED = "failed"  # 실패(선택)
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_SENT, "Sent"),
        (STATUS_FAILED, "Failed"),
    ]

    request = models.ForeignKey(CampaignPayoutRequest, on_delete=models.CASCADE, related_name="lines")
    recipient_wallet = models.CharField(max_length=64, db_index=True)
    amount_sol = models.DecimalField(max_digits=18, decimal_places=9)

    # 어떤 submission 때문에 들어간 건지 추적(강추)
    submission = models.ForeignKey(CampaignSubmission, on_delete=models.SET_NULL, null=True, blank=True)

    # ✅ NEW: 지급 마킹/추적
    payout_status = models.CharField(
        max_length=16,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
        db_index=True,
    )
    sent_at = models.DateTimeField(null=True, blank=True)
    payout_tx_signature = models.CharField(max_length=128, blank=True, default="", db_index=True)
    admin_note = models.CharField(max_length=255, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["recipient_wallet"]),
            models.Index(fields=["payout_status"]),
        ]
        constraints = [
            # submission이 NULL이면 여러 줄 가능, 값이 있으면 “한 submission당 1줄”만
            models.UniqueConstraint(
                fields=["submission"],
                name="uniq_payoutline_per_submission",
                condition=models.Q(submission__isnull=False),
            ),
        ]

    def __str__(self):
        return f"{self.recipient_wallet[:6]}... {self.amount_sol} SOL ({self.payout_status})"

class UserAttendance(models.Model):
    """
    출석 체크 (UTC date 기준)
    - 유저당 하루 1개만 생성
    - 생성되면 1 포인트 지급 (view에서 처리)
    """
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="attendances")
    date_utc = models.DateField(db_index=True)  # ✅ 00:00 UTC 기준 "오늘" 날짜
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [("user", "date_utc")]
        indexes = [
            models.Index(fields=["user", "-date_utc"]),
        ]
        ordering = ["-date_utc"]

    def __str__(self):
        return f"{self.user.wallet_address[:6]}... {self.date_utc}"


class Announcement(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=220, unique=True, blank=True)
    content = RichTextUploadingField()  # ✅ 이미지/링크 포함 리치 텍스트
    is_published = models.BooleanField(default=True)
    pinned = models.BooleanField(default=False)

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    published_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-pinned", "-published_at", "-created_at"]

    def save(self, *args, **kwargs):
        if not self.slug:
            base = slugify(self.title)[:200] or "announcement"
            slug = base
            i = 2
            while Announcement.objects.filter(slug=slug).exclude(pk=self.pk).exists():
                slug = f"{base}-{i}"
                i += 1
            self.slug = slug

        # publish 시간 자동
        if self.is_published and self.published_at is None:
            self.published_at = timezone.now()

        super().save(*args, **kwargs)

    def __str__(self):
        return self.title

class MarketStudyPost(models.Model):
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=220, unique=True)
    excerpt = models.CharField(max_length=280, blank=True, default="")
    content = models.TextField(blank=True, default="")
    is_free = models.BooleanField(default=False)  # ✅ 첫 글만 True로 두면 됨
    is_published = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title

class MarketStudySubscription(models.Model):
    ASSET_SOL = "sol"
    ASSET_LHX = "lhx"
    ASSET_CHOICES = [
        (ASSET_SOL, "SOL"),
        (ASSET_LHX, "LHX"),
    ]

    STATUS_PENDING = "pending"
    STATUS_VERIFIED = "verified"
    STATUS_REJECTED = "rejected"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_VERIFIED, "Verified"),
        (STATUS_REJECTED, "Rejected"),
    ]

    wallet_address = models.CharField(max_length=64, db_index=True)
    asset = models.CharField(max_length=8, choices=ASSET_CHOICES)
    amount = models.CharField(max_length=32, blank=True, default="")  # "0.3" or "100000"
    tx_signature = models.CharField(max_length=128, unique=True, db_index=True)

    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING)
    verified_at = models.DateTimeField(null=True, blank=True)
    expires_at = models.DateTimeField(null=True, blank=True, db_index=True)

    note = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["wallet_address", "-expires_at"]),
            models.Index(fields=["status", "-created_at"]),
        ]

    def is_active(self) -> bool:
        return bool(self.status == self.STATUS_VERIFIED and self.expires_at and self.expires_at > timezone.now())

    def __str__(self):
        return f"{self.wallet_address} {self.asset} {self.status}"

# ============================================================
# Hourly Crypto Predictions (cached in DB, updated by Celery)
# ============================================================
class HourlyCryptoPrediction(models.Model):
    asset = models.CharField(max_length=8, unique=True, db_index=True)  # BTC, ETH, SOL
    verdict = models.CharField(max_length=8, blank=True, default="")     # "up" or "down"
    reasoning = models.TextField(blank=True, default="")
    as_of_utc = models.DateTimeField(null=True, blank=True, db_index=True)
    next_update_utc = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["asset"]

    def __str__(self):
        return f"{self.asset}: {self.verdict} (as of {self.as_of_utc})"
    
# ============================================================
# Games (NEW)
# ============================================================

class UserSolanaWallet(models.Model):
    """
    A logged-in user may register one or more Solana wallet addresses
    to be used as VALID bet senders for Games.

    - user_profile: the LinkHash login identity (can be a non-sol wallet)
    - solana_wallet: Solana address that must match the on-chain TX sender
    - is_primary: for UI convenience ("the" wallet to show)
    """

    user_profile = models.ForeignKey(
        UserProfile,
        on_delete=models.CASCADE,
        related_name="solana_wallets",
        db_index=True,
    )

    solana_wallet = models.CharField(
        max_length=64,
        unique=True,           # ✅ globally unique (one Solana wallet cannot belong to multiple users)
        db_index=True,
    )

    is_primary = models.BooleanField(default=True, db_index=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-is_primary", "-updated_at"]
        constraints = [
            # Prevent duplicate wallet per same user (unique=True already prevents global dupes,
            # but this is still nice for clarity / migrations)
            models.UniqueConstraint(
                fields=["user_profile", "solana_wallet"],
                name="uniq_user_solana_wallet_per_user",
            ),
            # ✅ only one primary per user
            models.UniqueConstraint(
                fields=["user_profile"],
                condition=models.Q(is_primary=True),
                name="uniq_primary_solana_wallet_per_user",
            ),
        ]

    def __str__(self):
        return f"{self.user_profile.wallet_address} -> {self.solana_wallet}"


class Game(models.Model):
    """
    Master definition of a game (e.g., Lottery Hourly, Lottery Daily, Lottery Daily, FOMO Limit, FOMO No-Limit).
    This is NOT a "round". Rounds are in GameSession.
    """


    # Top-level type
    TYPE_LOTTERY = "lottery"
    TYPE_FOMO = "fomo"
    TYPE_CHOICES = [
        (TYPE_LOTTERY, "Lottery"),
        (TYPE_FOMO, "FOMO"),
    ]

    # Subtypes / modes (optional, but useful for filtering)
    LOTTERY_HOURLY = "hourly"
    LOTTERY_DAILY = "daily"

    FOMO_LIMIT = "limit"
    FOMO_NO_LIMIT = "no_limit"

    subtype = models.CharField(
        max_length=16,
        blank=True,
        default="",
        help_text="lottery: hourly/daily, fomo: limit/no_limit",
        db_index=True,
    )

    category = models.CharField(max_length=16, choices=TYPE_CHOICES, db_index=True)
    name = models.CharField(max_length=80, db_index=True)

    description = models.TextField(blank=True, default="")

    # Betting / ticket rules (interpreted based on category/subtype)
    min_bet_sol = models.DecimalField(max_digits=18, decimal_places=9, default=0)   # e.g., fomo no-limit min
    fixed_bet_sol = models.DecimalField(max_digits=18, decimal_places=9, default=0) # e.g., fomo limit = 0.01
    increment_pct = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        default=0,
        help_text="No-limit: next bet must be prev * (1 + increment_pct/100). e.g. 3.00",
    )

    protocol_fee_pct = models.DecimalField(max_digits=6, decimal_places=2, default=8.00)

    # Lottery rule params
    # hourly: pick 2 out of 5; daily: pick 3 out of 10
    lottery_pick_count = models.PositiveSmallIntegerField(default=0)
    lottery_number_max = models.PositiveSmallIntegerField(default=0)
    lottery_rollover_enabled = models.BooleanField(default=False)

    # FOMO rule params
    fomo_timer_seconds = models.PositiveIntegerField(default=600)  # 10 minutes default

    # Solana wallets for this game (store as text for now)
    # IMPORTANT: Storing private keys in DB is dangerous; prefer encrypted storage or KMS.
    sol_public_key = models.CharField(max_length=64, blank=True, default="", db_index=True)
    sol_private_key = models.TextField(blank=True, default="")  # consider encrypting!

    is_active = models.BooleanField(default=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["category", "subtype", "is_active"]),
        ]

    def __str__(self):
        return f"{self.name} ({self.category}/{self.subtype})"


class GameSession(models.Model):
    """
    Each round of a game.
    - FOMO: one session is a pot that ends when timer hits 0.
    - Lottery: one session is an hourly/daily draw.
    """

    STATUS_OPEN = "open"
    STATUS_LOCKED = "locked"   # optional: no more bets, pending settlement
    STATUS_SETTLED = "settled"
    STATUS_CANCELED = "canceled"
    STATUS_CHOICES = [
        (STATUS_OPEN, "Open"),
        (STATUS_LOCKED, "Locked"),
        (STATUS_SETTLED, "Settled"),
        (STATUS_CANCELED, "Canceled"),
    ]

    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name="sessions", db_index=True)

    # optional round identifier you can show in UI
    round_no = models.PositiveIntegerField(default=1)

    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_OPEN, db_index=True)

    # Pot tracking
    pot_total_sol = models.DecimalField(max_digits=18, decimal_places=9, default=0)

    fee_pct_snapshot = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        default=8.00,
        help_text="Snapshot protocol fee for this session (in case game fee changes later).",
    )

    # FOMO-specific: when the session will end if no new bets
    fomo_ends_at = models.DateTimeField(null=True, blank=True, db_index=True)

    # Lottery-specific: draw time + winning numbers
    lottery_draw_at = models.DateTimeField(null=True, blank=True, db_index=True)
    winning_numbers = models.CharField(
        max_length=64,
        blank=True,
        default="",
        help_text='Store as "1-4" or "1-4-9" etc.',
    )

    # Rollover support (lottery)
    rollover_from = models.ForeignKey(
        "self",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="rollover_to_sessions",
        help_text="If this session's pot includes rollover from a prior session.",
    )

    # Settlement bookkeeping
    settled_at = models.DateTimeField(null=True, blank=True)
    settlement_tx_signature = models.CharField(max_length=128, blank=True, default="", db_index=True)

    # Celery scheduled settlement task id (used to revoke/reschedule on every bet)
    settle_task_id = models.CharField(max_length=255, blank=True, default="", db_index=True)

    created_at = models.DateTimeField(auto_now_add=True)


    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["game", "status", "-created_at"]),
            models.Index(fields=["game", "round_no"]),
            models.Index(fields=["lottery_draw_at"]),
            models.Index(fields=["fomo_ends_at"]),
        ]
        unique_together = [("game", "round_no")]

    def __str__(self):
        return f"Session {self.game.name} #{self.round_no} ({self.status})"


class Participation(models.Model):
    """
    Every bet/ticket purchase.
    IMPORTANT: Only logged-in users can bet, so link to UserProfile.
    """

    KIND_BET = "bet"
    KIND_TICKET = "ticket"
    KIND_CHOICES = [
        (KIND_BET, "Bet"),
        (KIND_TICKET, "Ticket"),
    ]

    STATUS_PENDING = "pending"   # user submitted tx signature, not verified yet
    STATUS_VERIFIED = "verified" # tx verified on-chain
    STATUS_REJECTED = "rejected" # invalid tx / wrong amount / etc
    STATUS_REFUNDED = "refunded"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_VERIFIED, "Verified"),
        (STATUS_REJECTED, "Rejected"),
        (STATUS_REFUNDED, "Refunded"),
    ]

    session = models.ForeignKey(GameSession, on_delete=models.CASCADE, related_name="participations", db_index=True)
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="participations", db_index=True)

    kind = models.CharField(max_length=16, choices=KIND_CHOICES, default=KIND_BET)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING, db_index=True)

    # The amount user claims they sent (SOL)
    amount_sol = models.DecimalField(max_digits=18, decimal_places=9)

    # On-chain proof provided by user
    tx_signature = models.CharField(max_length=128, unique=True, db_index=True)

    # Game-specific data
    # Lottery: store chosen numbers like "1-4" or "1-4-9"
    picked_numbers = models.CharField(max_length=64, blank=True, default="")

    # FOMO: ordering matters (king-of-hill). Store increasing index for fast queries.
    bet_index = models.PositiveIntegerField(default=0, db_index=True)

    # Optional debugging / admin notes
    reject_reason = models.CharField(max_length=255, blank=True, default="")
    verified_at = models.DateTimeField(null=True, blank=True)

    # ✅ On-chain block timestamp (UTC) — used for "limit" mode king ordering
    block_time = models.DateTimeField(null=True, blank=True, db_index=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["session", "status", "-created_at"]),
            models.Index(fields=["user_profile", "-created_at"]),
            models.Index(fields=["session", "bet_index"]),
            models.Index(fields=["session", "-block_time"]),
        ]

    def __str__(self):
        return f"{self.kind} {self.amount_sol} SOL ({self.status}) - {self.user_profile.wallet_address[:6]}..."

class InvalidBet(models.Model):
    """
    Stores any bet attempt that failed verification or game rules,
    so you can manually refund later.
    """

    # ---- why invalid ----
    REASON_LATE = "late"
    REASON_TOO_OLD = "too_old"
    REASON_BAD_RECEIVER = "bad_receiver"
    REASON_BAD_AMOUNT = "bad_amount"
    REASON_BAD_SENDER = "bad_sender"
    REASON_TX_NOT_FOUND = "tx_not_found"
    REASON_TX_FAILED = "tx_failed"
    REASON_DUPLICATE_TX = "duplicate_tx"
    REASON_UNKNOWN = "unknown"

    REASON_CHOICES = [
        (REASON_LATE, "Late (after timer)"),
        (REASON_TOO_OLD, "Too old (before timer window)"),
        (REASON_BAD_RECEIVER, "Wrong receiver wallet"),
        (REASON_BAD_AMOUNT, "Invalid amount"),
        (REASON_BAD_SENDER, "Sender wallet not registered"),
        (REASON_TX_NOT_FOUND, "Transaction not found"),
        (REASON_TX_FAILED, "Transaction failed"),
        (REASON_DUPLICATE_TX, "Duplicate TX"),
        (REASON_UNKNOWN, "Unknown"),
    ]

    # ---- refund workflow ----
    REFUND_PENDING = "pending"
    REFUND_REFUNDED = "refunded"
    REFUND_IGNORED = "ignored"  # admin decided: no refund required / already handled elsewhere

    REFUND_STATUS_CHOICES = [
        (REFUND_PENDING, "Pending"),
        (REFUND_REFUNDED, "Refunded"),
        (REFUND_IGNORED, "Ignored"),
    ]

    game = models.ForeignKey(Game, on_delete=models.CASCADE, related_name="invalid_bets", db_index=True)
    session = models.ForeignKey(
        GameSession,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="invalid_bets",
        db_index=True,
    )

    submitter_wallet = models.CharField(max_length=64, db_index=True, blank=True, default="")
    receiver_wallet = models.CharField(max_length=64, db_index=True, blank=True, default="")

    tx_signature = models.CharField(max_length=128, db_index=True)
    amount_onchain_sol = models.DecimalField(max_digits=18, decimal_places=9, null=True, blank=True)
    block_time = models.DateTimeField(null=True, blank=True)

    reason = models.CharField(max_length=32, choices=REASON_CHOICES, default=REASON_UNKNOWN, db_index=True)
    note = models.TextField(blank=True, default="")

    # refund tracking
    refund_status = models.CharField(
        max_length=16,
        choices=REFUND_STATUS_CHOICES,
        default=REFUND_PENDING,
        db_index=True,
    )
    refund_tx_signature = models.CharField(max_length=128, blank=True, default="", db_index=True)
    refunded_at = models.DateTimeField(null=True, blank=True)
    admin_note = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["game", "session", "-created_at"]),
            models.Index(fields=["submitter_wallet", "-created_at"]),
            models.Index(fields=["reason", "refund_status", "-created_at"]),
        ]

    def __str__(self):
        return f"InvalidBet {self.reason} {self.tx_signature[:8]}... ({self.refund_status})"


class Winner(models.Model):
    """
    Actual winner records (can be multiple for lottery, usually 1 for FOMO).
    Payout is tracked separately so you can settle later.
    """

    PAYOUT_PENDING = "pending"
    PAYOUT_SENT = "sent"
    PAYOUT_FAILED = "failed"
    PAYOUT_CHOICES = [
        (PAYOUT_PENDING, "Pending"),
        (PAYOUT_SENT, "Sent"),
        (PAYOUT_FAILED, "Failed"),
    ]

    session = models.ForeignKey(GameSession, on_delete=models.CASCADE, related_name="winners", db_index=True)
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="wins", db_index=True)

    # link to their participation that caused the win (optional but very useful)
    participation = models.ForeignKey(
        Participation,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="winner_records",
        help_text="The bet/ticket that made them a winner (for FOMO, last bet; for lottery, the winning ticket).",
    )

    payout_amount_sol = models.DecimalField(max_digits=18, decimal_places=9, default=0)
    payout_status = models.CharField(max_length=16, choices=PAYOUT_CHOICES, default=PAYOUT_PENDING, db_index=True)

    payout_tx_signature = models.CharField(max_length=128, blank=True, default="", db_index=True)
    paid_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["session", "-created_at"]),
            models.Index(fields=["user_profile", "-created_at"]),
            models.Index(fields=["payout_status", "-created_at"]),
        ]

    def __str__(self):
        return f"Winner {self.user_profile.wallet_address[:6]}... {self.payout_amount_sol} SOL ({self.payout_status})"
