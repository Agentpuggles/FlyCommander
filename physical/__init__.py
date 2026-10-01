"""FlyCommander physical-table mode.

Hybrid architecture: camera vision proposes, explicit player events dispose.
The authoritative `PhysicalGameState` is the only thing the Fly brain ever
sees, and only through strictly public observations.
"""
