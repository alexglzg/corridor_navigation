from linemap import LineMap

# FILENAME = "Corridor-MapDataset/testmap1.png"
# FILENAME = "Corridor-MapDataset/testmap5.png"
# FILENAME = "Corridor-MapDataset/Interesting Maps/map_7.png"
# FILENAME = "Corridor-MapDataset/Interesting Maps/map_25.png"
# FILENAME = "Corridor-MapDataset/Interesting Maps/map_6.png"
# FILENAME = "Corridor-MapDataset/Scale Maps/Larger Scale/structured_map_20.pgm"
# FILENAME = "Corridor-MapDataset/Scale Maps/Larger Scale/structured_rows_20.pgm"
# FILENAME = "Corridor-MapDataset/Scale Maps/Small Scale/room_map_2.pgm"
# FILENAME = "Corridor-MapDataset/Scale Maps/Small Scale/room_map_10.pgm"
# FILENAME = "Corridor-MapDataset/Scale Maps/Small Scale/few_rows.png"
# FILENAME = "Corridor-MapDataset/Scale Maps/Small Scale/structured_map_2.pgm"
# FILENAME = "Corridor-MapDataset/Scale Maps/Small Scale/structured_rows.pgm"
# FILENAME = "Corridor-MapDataset/Real Maps/gazebo_map/map.pgm"
# FILENAME = "Corridor-MapDataset/Interesting Maps/populated_grid_map.pgm"

FILENAME = "Corridor-MapDataset/Interesting Maps/gmap_.pgm"

# map_5 misses a door
# map_8 misses subpart of a room
# map_17 has diagonals

THRESHOLD = 250  # PNG

DEBUG = 1
SHOW = 1
SAVE = 0


def main():
    path = f'{FILENAME}'

    floor_plan = LineMap(path, threshold=THRESHOLD, debug=DEBUG)
    # floor_plan = LineMap(path, threshold=THRESHOLD, debug=DEBUG, resolution=0.05)

    # floor_plan.process(structured=False, expect_obstacles=False)
    # floor_plan.process(structured=False, expect_obstacles=True)
    # floor_plan.process(structured=True)
    # floor_plan.process_structured(extend_obstacles=False)
    # floor_plan.process_structured(extend_obstacles=True)
    # floor_plan.process_line_pairing()
    floor_plan.process_slicing()

    floor_plan.print(filename="out", show=SHOW, save=SAVE, debug=0,
                     mode=["rectangles", "lines", "snap_points"])
    floor_plan.print(filename="in", show=0, save=SAVE, debug=0, mode=[])
    # floor_plan.draw_graph()


if __name__ == '__main__':
    main()
