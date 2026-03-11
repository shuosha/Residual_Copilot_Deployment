"""PolicyAgent — autonomous policy rollout and trajectory replay action agent."""

import time
import multiprocess as mp

from robot_control.agents.action_agent import ActionAgent


class PolicyAgent(ActionAgent):

    def __init__(
        self,
        action_receiver="policy",
        **kwargs,
    ) -> None:
        assert action_receiver in ("policy", "replay"), \
            f"PolicyAgent only supports policy/replay, got {action_receiver}"
        super().__init__(**kwargs)
        self.action_receiver = action_receiver
        self.reset = mp.Value('b', False)

        # Extend key states
        self.key_states.update({
            "r": False,  # reset
            "s": False,  # start
        })

    def _init_listeners(self):
        pass  # main process writes commands directly

    def _process_command(self) -> list:
        self._handle_record_keys()

        if self.key_states["r"]:
            self.reset.value = True
            self.record_stop.value = True
        if self.key_states["s"]:
            self.reset.value = False
            self.record_start.value = True

        return list(self.command)

    def _cleanup_listeners(self):
        pass
