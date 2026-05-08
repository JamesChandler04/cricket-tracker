from enum import Enum
from dataclasses import dataclass
import math

class Key(Enum):
    esc = 27
    q = ord('q')
    c = ord('c')
    space = ord(' ')
    a = ord('a')
    d = ord('d')
    s = ord('s')
    o = ord('o')
    r = ord('r')
    t = ord('t')
    A = ord('A')
    D = ord('D')
    f = ord('f')
    b = ord('b')

@dataclass
class Coord:
    x: int
    y: int

    def __str__(self):
        return f"({self.x}, {self.y})"

@dataclass
class BallData:
    top_left: Coord
    bottom_right: Coord
    centre: Coord
    seam_start: Coord
    seam_end: Coord
    seam_angle: float

    def __str__(self):
        return (f"BallData(top_left={self.top_left}, "
                f"bottom_right={self.bottom_right}, "
                f"centre={self.centre}, "
                f"seam_start={self.seam_start}, "
                f"seam_end={self.seam_end}, "
                f"seam_angle={self.seam_angle:.2f} degrees)")

    def calc_seam_angle(self):
        self.seam_angle = math.degrees(math.atan2(self.seam_end.y - self.seam_start.y, self.seam_end.x - self.seam_start.x))

    def calc_centre(self):
        self.centre = Coord(
            x=(self.top_left.x + self.bottom_right.x) // 2,
            y=(self.top_left.y + self.bottom_right.y) // 2
        )