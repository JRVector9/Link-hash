# market/urls.py
from django.urls import path
from .views import (
    # core
    home, about, partnership,

    # auth
    phantom_nonce, phantom_verify, logout_view,

    # account
    profile, settings_page,

    # partnership
    submit_ad,

    # public campaigns
    campaign_list, campaign_detail, campaign_submit,
    campaign_submissions_partial,

    # campaign create
    campaign_create, campaign_create_verify,

    # creator dashboard
    creator_campaign_list,
    creator_campaign_manage,
    creator_mark_submission,
    creator_request_payout,

    # leaderboard
    leaderboard,
    public_profile,

    # announcements
    announcement_list,
    announcement_detail,

    # prediction+
    prediction_plus,
    api_hourly_predictions,
    api_market_study_subscribe,
    market_study_detail,

    games_home, lottery_detail, fomo_detail,

    game_session_state_json,
    game_place_bet,
    user_hud_state_json,
    user_update_solana_wallet
)

urlpatterns = [
    # =========================
    # Core
    # =========================
    path("", home, name="home"),
    path("about/", about, name="about"),
    path("partnership/", partnership, name="partnership"),

    # =========================
    # Auth (Phantom)
    # =========================
    path("auth/phantom/nonce/", phantom_nonce, name="phantom_nonce"),
    path("auth/phantom/verify/", phantom_verify, name="phantom_verify"),
    path("logout/", logout_view, name="logout"),

    # =========================
    # Account
    # =========================
    path("profile/", profile, name="profile"),
    path("settings/", settings_page, name="settings"),

    # =========================
    # Partnership
    # =========================
    path("partnership/submit-ad/", submit_ad, name="submit_ad"),

    # =========================
    # Public Campaigns
    # =========================
    path("campaigns/", campaign_list, name="campaign_list"),
    path("campaigns/<int:campaign_id>/", campaign_detail, name="campaign_detail"),
    path("campaigns/<int:campaign_id>/submit/", campaign_submit, name="campaign_submit"),
    path(
        "campaigns/<int:campaign_id>/submissions/",
        campaign_submissions_partial,
        name="campaign_submissions_partial",
    ),

    # =========================
    # Campaign Create
    # =========================
    path("campaigns/create/", campaign_create, name="campaign_create"),
    path("campaigns/create/verify/", campaign_create_verify, name="campaign_create_verify"),

    # =========================
    # Creator Dashboard (NEW)
    # =========================
    path(
        "creator/campaigns/",
        creator_campaign_list,
        name="creator_campaign_list",
    ),
    path(
        "creator/campaigns/<int:campaign_id>/",
        creator_campaign_manage,
        name="creator_campaign_manage",
    ),
    path(
        "creator/campaigns/<int:campaign_id>/mark/",
        creator_mark_submission,
        name="creator_mark_submission",
    ),
    path(
        "creator/campaigns/<int:campaign_id>/payout/",
        creator_request_payout,
        name="creator_request_payout",
    ),

    # =========================
    # Leaderboard
    # =========================
    path(
        "leaderboard/",
        leaderboard,
        name="leaderboard",
    ),
    path(
        "u/<str:wallet>/",
        public_profile,
        name="public_profile",
    ),
    path("announcements/",
         announcement_list,
         name="announcement_list"),
    path("announcements/<slug:slug>/",
         announcement_detail,
         name="announcement_detail"),

    # =========================
    # Prediction+
    # =========================
    path("prediction-plus/",
         prediction_plus,
         name="prediction_plus"),
    path("api/predictions/hourly/",
         api_hourly_predictions,
         name="api_hourly_predictions"),
    path("api/market-study/subscribe/",
         api_market_study_subscribe,
         name="api_market_study_subscribe"),
    path("market-study/<slug:slug>/",
         market_study_detail,
         name="market_study_detail"),

    # Games
    path("games/", games_home, name="games_home"),
    path("games/lottery/<int:game_id>/", lottery_detail, name="lottery_detail"),
    path("games/fomo/<int:game_id>/", fomo_detail, name="fomo_detail"),

    # ✅ Modal APIs (no page navigation)
    path("api/games/<int:game_id>/state/", game_session_state_json, name="game_session_state_json"),
    path("api/games/<int:game_id>/bet/", game_place_bet, name="game_place_bet"),
    path("api/hud/state/", user_hud_state_json, name="user_hud_state_json"),
    path("games/solana-wallet/", user_update_solana_wallet, name="user_update_solana_wallet"),


]
