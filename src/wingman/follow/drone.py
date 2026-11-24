"""Where movement commands go. The follow loop only calls send_up_speed(), hold() and
step(), so SimDrone (Layer 1) and PX4Drone (SITL / real drone) are interchangeable.

Convention: positive speed = UP, in metres per second.
"""
import asyncio
from collections import deque


class SimDrone:
    """A pretend drone that obeys up/down speed commands like a real one would:
    commands arrive late (latency), speed changes gradually (response time and an
    acceleration limit), and there is a top speed and height limits."""

    def __init__(self, cfg, start_x, start_y):
        self.x, self.y, self.vy = start_x, start_y, 0.0
        self.t = 0.0
        self._cmd = 0.0
        self._pending = deque()  # (time the command takes effect, speed)
        self.latency = cfg.loop.latency_s
        self.response = cfg.sim.response_time_s
        self.max_speed = cfg.controller.max_speed_mps
        self.max_accel = cfg.controller.max_accel_mps2
        self.y_min, self.y_max = cfg.sim.min_view_center_m, cfg.sim.max_view_center_m

    def send_up_speed(self, up_mps):
        self._pending.append((self.t + self.latency, up_mps))

    def hold(self):
        self.send_up_speed(0.0)

    def step(self, dt):
        """Advance the simulation by dt seconds."""
        self.t += dt
        while self._pending and self._pending[0][0] <= self.t + 1e-9:
            self._cmd = self._pending.popleft()[1]
        target = max(-self.max_speed, min(self.max_speed, self._cmd))
        change = (target - self.vy) * min(1.0, dt / self.response)
        limit = self.max_accel * dt
        self.vy += max(-limit, min(limit, change))
        self.y += self.vy * dt
        if not self.y_min <= self.y <= self.y_max:  # hit a height limit: stop there
            self.y = max(self.y_min, min(self.y_max, self.y))
            self.vy = 0.0


class PX4Drone:
    """PX4 (SITL or the real Pixhawk) through MAVSDK. NOT TESTED - needs PX4 SITL.

    Intended use: the pilot takes off and hovers manually; then start_offboard() hands
    control to the Jetson. The pilot can switch the flight mode on the remote at any
    time to take back control; PX4's offboard-loss failsafe (set in QGroundControl)
    decides what happens if this program stops sending commands.
    PX4 uses "down" as positive, so an UP speed is sent as a negative down speed.
    """

    def __init__(self, address):
        from mavsdk import System
        self._loop = asyncio.new_event_loop()
        self._sys = System()
        self._run(self._sys.connect(system_address=address))
        self._run(self._wait_connected())

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    async def _wait_connected(self):
        async for state in self._sys.core.connection_state():
            if state.is_connected:
                return

    def height_m(self):
        async def first():
            async for pv in self._sys.telemetry.position_velocity_ned():
                return -pv.position.down_m
        return self._run(first())

    def start_offboard(self):
        from mavsdk.offboard import VelocityBodyYawspeed
        # PX4 requires a setpoint to be streaming before offboard mode will start.
        self._run(self._sys.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)))
        self._run(self._sys.offboard.start())

    def send_up_speed(self, up_mps):
        from mavsdk.offboard import VelocityBodyYawspeed
        self._run(self._sys.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, -up_mps, 0.0)))

    def hold(self):
        self.send_up_speed(0.0)

    def step(self, dt):
        pass  # real time passes by itself

    def stop_offboard(self):
        self._run(self._sys.offboard.stop())
