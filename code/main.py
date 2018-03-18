import argparse
import copy
import math
import operator
import os
from collections import deque

import numpy as np
import pandas as pd
from scipy import stats


def read_data(fname):
    with open(os.path.join(os.path.dirname(__file__), "../data/{}".format(fname))) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    return content


class Point():
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)


def distance(pointOne, pointTwo):
    dist = math.fabs(pointOne.x - pointTwo.x) + \
           math.fabs(pointOne.y - pointTwo.y)
    return int(dist)


class Trip:
    def __init__(self, start_x, start_y,
                 end_x, end_y,
                 earliest_start, latest_finish, count):
        self.start = Point(start_x, start_y)
        self.end = Point(end_x, end_y)
        self._dist = distance(self.start, self.end)
        self._earliest_start = int(earliest_start)
        self.latest_finish = int(latest_finish)
        time_diff = self.latest_finish - self.earliest_start
        self._score = time_diff - self._dist
        self._count = count

    @property
    def earliest_start(self):
        return self._earliest_start

    @property
    def end_point(self):
        return self.end

    @property
    def distance(self):
        return self._dist

    @property
    def count(self):
        return self._count

    @property
    def score(self):
        return self._score


class potential_journey:
    def __init__(self, current_loc, ride):
        self.current_loc = current_loc
        self.ride = ride
        self.dist_to_pick_up = self.dist_to_trip(current_loc, ride.end_point)
        self.ride_score = ride.score

    @staticmethod
    def dist_to_trip(current_loc, end_point):
        return distance(current_loc, end_point)

    @property
    def final_score(self):
        return self.normalized_dist_to_pick_up * self.normalized_ride_score

    def normalized_ride_score(self, norm):
        self.normalized_ride_score = self.ride_score / norm

    def normalized_dist_to_pick_up(self, norm):
        self.normalized_dist_to_pick_up = self.ride_score / norm


def check_rider_can_reach_dest_in_time(current_time, end_time, current_loc, trip_time):
    dist = distance(current_loc, trip_time)
    remaining_time = end_time - current_time
    if dist > remaining_time:
        return False
    else:
        return True


def accept_ride_and_update_state(state_dataFrame, next_ride, row, current_loc):
    rides_taken = state_dataFrame.at[row, "Rides"]
    state_dataFrame.at[row, "Rides"] = "{} {}".format(rides_taken, next_ride.count)

    trip_count = state_dataFrame.at[row, "Trip_Count"]
    state_dataFrame.at[row, "Trip_Count"] = trip_count + 1

    journey_dist = distance(current_loc, next_ride.end_point)
    state_dataFrame.at[row, "TimeSteps_To_Dest"] = journey_dist
    state_dataFrame.at[row, "EndCoordinate"] = next_ride.end_point

    return state_dataFrame


def main(filename):
    content = read_data(filename)
    rows, columns, number_cars, number_rides, bonus, Total_time = [int(x) for x in content[0].split(" ")]
    data = [item.split(" ") for item in content[1:]]

    Rides = []
    count = 0

    for row in data:
        Rides.append(Trip(*row, count))
        count += 1

    rides_queue = deque(Rides)

    slice_rides = [int(ride.distance) for ride in rides_queue]

    a = np.array(slice_rides)
    print("Printing Ride Length Stats:\n", stats.describe(a))

    state_dataFrame = pd.DataFrame(data={"TimeSteps_To_Dest": [0] * number_cars,
                                         "Vechicle ID": [r for r in range(0, number_cars, 1)],
                                         "EndCoordinate": [Point(0, 0)] * number_cars,
                                         "Trip_Count": [0] * number_cars,
                                         "Rides": [""] * number_cars})

    state_dataFrame.set_index(["Vechicle ID"], inplace=True, verify_integrity=True)

    Time = 0

    while Time < Total_time:
        for i, row in state_dataFrame.iterrows():
            if row.get("TimeSteps_To_Dest") <= 0:
                # Simply pick the next passenger in the priority Queue
                if len(rides_queue) == 0:
                    print("All rides done")
                    return submit_data(state_dataFrame, filename)

                # Current location is end of previous ride
                current_loc = state_dataFrame.at[i, "EndCoordinate"]

                # Copy of the remaining rides
                remanining_rides = copy.copy(rides_queue)
                valid_journeys = []
                try:
                    while True:
                        ride = remanining_rides.popleft()
                        if check_rider_can_reach_dest_in_time(Time, Total_time, current_loc, ride.end_point):
                            valid_journeys.append(potential_journey(current_loc, ride))
                except:
                    IndexError
                    # No op all rides checked

                if len(valid_journeys) > 0:
                    norm = np.linalg.norm([jour.dist_to_pick_up for jour in valid_journeys])
                    [ride.normalized_dist_to_pick_up(norm) for ride in valid_journeys]

                    norm = np.linalg.norm([jour.ride_score for jour in valid_journeys])
                    [ride.normalized_ride_score(norm) for ride in valid_journeys]

                    key = operator.attrgetter("final_score")
                    valid_journeys.sort(key=key)

                    chosen_journey = valid_journeys.pop()
                    state_dataFrame = accept_ride_and_update_state(state_dataFrame, chosen_journey.ride, i, current_loc)
                    rides_queue.remove(chosen_journey.ride)

        time_increment = max(1, min(state_dataFrame["TimeSteps_To_Dest"]))
        Time += time_increment
        print("Time done", Time / Total_time)

        # Once this loop is over every Vechicle has its first ride
        state_dataFrame.update(state_dataFrame["TimeSteps_To_Dest"].map(lambda x: x - time_increment))

    else:
        print("Time is UP")

    submit_data(state_dataFrame, filename)


def submit_data(state_dataFrame, filename):
    output_file = "{}_output.txt".format(filename.split(".in")[0])
    file_lines = []
    for i, row in state_dataFrame.iterrows():
        file_lines.append('{}{}\n'.format(str(state_dataFrame.at[i, "Trip_Count"]),
                                          state_dataFrame.at[i, "Rides"]))
    with open(output_file, mode='w') as f:
        f.writelines(file_lines)
    f.close()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run City Simulation for input')
    parser.add_argument('--file', type=str, default="",
                        help='Pass in Input File Name ')
    args = parser.parse_args()
    if args.file != "":
        main(args.file)
    else:
        data_File = ["a_example.in", "b_should_be_easy.in", "c_no_hurry.in",
                     "d_metropolis.in", "e_high_bonus.in"]

        for file in data_File:
            print("=====Starting: {}=====".format(file))
            main(file)
