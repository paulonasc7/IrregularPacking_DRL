# -*- coding: utf-8 -*-

import numpy as np


class HM_heuristic(object):
    def __init__(self, c=0.01):
        self.c = c

    def pack(self, env, item_id):
        """Brute-force heightmap heuristic placement for one loaded body."""
        Hc = env.box_heightmap()
        if item_id not in env.loaded_ids:
            return False, None

        best = None
        box_w, box_h = env.resolution, env.resolution

        for roll in np.arange(0, 2 * np.pi, np.pi / 2):
            for pitch in np.arange(0, 2 * np.pi, np.pi / 2):
                for yaw in np.arange(0, 2 * np.pi, np.pi / 2):
                    trans = np.array([roll, pitch, yaw], dtype=np.float64)
                    Ht, Hb = env.item_hm(item_id, trans)
                    w, h = Ht.shape
                    if w <= 0 or h <= 0 or w > box_w or h > box_h:
                        continue

                    for x in range(0, box_w - w + 1):
                        for y in range(0, box_h - h + 1):
                            z = np.max(Hc[x:x + w, y:y + h] - Hb)
                            updated = np.maximum((Ht > 0) * (Ht + z), Hc[x:x + w, y:y + h])
                            if np.max(updated) > env.box_size[2]:
                                continue

                            score = self.c * (x + y) + np.sum(Hc) + np.sum(updated) - np.sum(Hc[x:x + w, y:y + h])
                            cand = np.array([roll, pitch, yaw, x, y, z, score], dtype=np.float64)
                            if best is None or cand[6] < best[6]:
                                best = cand

        if best is None:
            return False, None

        success = env.pack_item(item_id, best[:6].copy())
        return bool(success), best
