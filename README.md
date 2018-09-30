# nba-schedule-creator
Simple proof of concept to demonstrate how to create a schedule for an NBA season using simulated annealing

## Usage
The code was written for python 3, and uses the [geopy](https://github.com/geopy/geopy) package. Run script using
```
python schedule.py
```

## Details
The script creates a reasonably valid schedule for the 2018/2019 NBA season. When creating the actual NBA schedule, they use ~1200 weights/penalties. This script only factors in around 4 aspects: 
- No games on the same day (strongest penalty)
- Minimal number of back-to-backs/maximizing rest
- Minimum distance travelled
- No long road trips

This obviously limits how realistic the schedule is --- it does not care about arena availabilities, holidays, marquee matchups, arena elevations (e.g. Denver), All Star break, having equally hard schedules for all teams etc. --- otherwise the results are quite believable. Now it is only a question of adding more relevant features/penalties.

One weird property of the NBA schedule is that the teams that share conferences but are in different divisions some years play 3 times, other years 4 times, which is defined by a rotating schedule that repeats itself every 5 years. I was not able to find any source online which would describe this in detail, therefore opted to randomly assign the teams that play 3 times while making sure the schedule is valid (every team plays 82 games, and the same amount at home/away). 
