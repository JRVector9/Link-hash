from django.core.management.base import BaseCommand
from django.db import transaction
from django.db.models import Count

from market.models import CampaignPayoutLine


class Command(BaseCommand):
    help = "Remove duplicate CampaignPayoutLine rows that share the same submission_id, keeping the newest one."

    def handle(self, *args, **options):
        # submission_id가 NULL인 라인은 unique 대상이 아니니 건드리지 않음
        dups = (
            CampaignPayoutLine.objects
            .exclude(submission__isnull=True)
            .values("submission_id")
            .annotate(cnt=Count("id"))
            .filter(cnt__gt=1)
        )

        if not dups.exists():
            self.stdout.write(self.style.SUCCESS("No duplicate payout lines found."))
            return

        total_deleted = 0

        with transaction.atomic():
            for row in dups:
                sid = row["submission_id"]
                # 최신 것 1개만 남기고 나머지 삭제
                ids = list(
                    CampaignPayoutLine.objects
                    .filter(submission_id=sid)
                    .order_by("-created_at", "-id")
                    .values_list("id", flat=True)
                )
                keep = ids[0]
                to_delete = ids[1:]
                deleted, _ = CampaignPayoutLine.objects.filter(id__in=to_delete).delete()
                total_deleted += deleted

                self.stdout.write(f"submission_id={sid} keep={keep} delete={len(to_delete)}")

        self.stdout.write(self.style.SUCCESS(f"Done. Deleted {total_deleted} duplicate payout lines."))
