import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer

log = logging.getLogger(__name__)

class GameConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.game_id = self.scope["url_route"]["kwargs"]["game_id"]
        self.group_name = f"game_{self.game_id}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def game_update(self, event):
        """
        Handles group_send(..., {"type": "game_update", "payload": {...}})
        """
        payload = event.get("payload")
        if not payload:
            return

        try:
            await self.send(text_data=json.dumps(payload))
        except Exception as e:
            log.exception("WS send failed (game_update) game_id=%s: %s", self.game_id, e)

    async def past_result(self, event):
        """
        Handles group_send(..., {"type": "past_result", "payload": {...}})
        Winner-only event used to prepend a past-results table row.
        """
        payload = event.get("payload")
        if not payload:
            return

        try:
            await self.send(text_data=json.dumps(payload))
        except Exception as e:
            log.exception("WS send failed (past_result) game_id=%s: %s", self.game_id, e)
