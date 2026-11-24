"""Model 1: keep the selected climber centred in the drone camera (runs on the Jetson).

Pieces (each swappable):
  camera.py      where pictures come from (simulated drone view, or the real camera)
  detector.py    find people, keep the lock on the chosen climber
  controller.py  "how far off-centre" -> gentle up/down speed
  drone.py       where commands go (simulated drone, or PX4 through MAVSDK)
  run_follow.py  the loop that connects them, plus simulation scoring
Settings: configs/follow.yaml
"""
