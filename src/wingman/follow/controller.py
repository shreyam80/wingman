"""Turn "how far is the climber from where we want them in the picture" into an
up/down speed. A PID controller:
  P (proportional): further off  -> move faster
  I (integral):     off for a long time -> push a bit more (off by default)
  D (derivative):   offset shrinking fast -> ease off, to avoid overshooting
plus a dead zone (ignore tiny offsets so the drone doesn't twitch) and speed and
acceleration limits.
"""


class VerticalController:
    def __init__(self, cfg, view_height_m):
        c = cfg.controller
        self.target, self.deadband = c.target_point_frac, c.deadband_frac
        self.kp, self.ki, self.kd = c.kp, c.ki, c.kd
        self.max_speed, self.max_accel = c.max_speed_mps, c.max_accel_mps2
        self.view_h = view_height_m
        self.reset()

    def reset(self):
        self.integral, self.prev_err, self.prev_out = 0.0, None, 0.0

    def update(self, climber_v_frac, dt):
        """climber_v_frac: climber's box centre height in the picture (0 = top edge,
        1 = bottom), or None if there is no locked climber. Returns UP speed in m/s."""
        if climber_v_frac is None:
            self.reset()
            return 0.0  # no target -> hold position
        err_frac = self.target - climber_v_frac  # > 0: climber is above where we want them
        if abs(err_frac) < self.deadband:
            err_frac = 0.0
        err_m = err_frac * self.view_h
        self.integral = max(-1.0, min(1.0, self.integral + err_m * dt))  # anti wind-up
        deriv = 0.0 if self.prev_err is None else (err_m - self.prev_err) / dt
        self.prev_err = err_m
        out = self.kp * err_m + self.ki * self.integral + self.kd * deriv
        out = max(-self.max_speed, min(self.max_speed, out))
        step = self.max_accel * dt  # don't change speed abruptly
        out = max(self.prev_out - step, min(self.prev_out + step, out))
        self.prev_out = out
        return out
