import pickle
import random
from collections import defaultdict
import math
from datetime import datetime, timedelta

import geopy
from geopy import distance
from geopy.geocoders import Nominatim

# use your own user agent
geolocator = Nominatim(user_agent="bfabiandev")

DAYS = 177
DISTANCES = dict()


# Class to represent teams, which contains the basic information (name, location, conference, division) and its schedule
# Also contains the weight for penalties that define what sort of schedule we get in the end
class Team:
    def __init__(self, name, location, conference, division):
        self.name = name
        self.loc = location
        self.conf = conference
        self.div = division
        self.schedule = defaultdict(set)

    def __str__(self):
        return self.name

    def add_game(self, game):
        self.schedule[game.date].add(game)

    def remove_game(self, game):
        self.schedule[game.date].remove(game)

    def loss(self):
        loss = 0

        current_home_team = self
        days_rest = 100
        road_trip_length = 0

        for i in range(1, DAYS+1):
            days_rest += 1
            if len(self.schedule[i]) == 0:
                continue
            else:
                if len(self.schedule[i]) > 1:
                    loss += 1e10 * len(self.schedule[i])

                game = next(iter(self.schedule[i]))
                next_home_team = game.home

                # get penalty based on distance flown
                dist = dist_teams(current_home_team, next_home_team)
                current_home_team = next_home_team
                loss += dist * 0.05

                # get penalty for back-to-backs
                if days_rest == 1:
                    rest_penalty = 200
                elif days_rest == 2:
                    rest_penalty = 50
                else:
                    rest_penalty = 0
                loss += rest_penalty

                # get penalty for long road-trips
                if game.home == self:
                    road_trip_length = 0
                else:
                    road_trip_length += 1

                if road_trip_length > 2:
                    road_trip_penalty = 10 * road_trip_length
                else:
                    road_trip_penalty = 0

                loss += road_trip_penalty

                days_rest = 0

        return loss

    def number_of_home_games(self):
        return sum([1 for date in self.schedule for game in self.schedule[date] if game.home == self])

    def number_of_back_to_backs(self):
        back_to_backs = 0
        for date1 in range(1, DAYS):
            if len(self.schedule[date1]) > 0 and len(self.schedule[date1 + 1]) > 0:
                back_to_backs += 1
        return back_to_backs

    def number_of_back_to_backs_to_backs(self):
        back_to_backs_to_backs = 0
        for date1 in range(1, DAYS-1):
            if len(self.schedule[date1]) > 0 and len(self.schedule[date1 + 1]) > 0 and len(self.schedule[date1 + 2]) > 0:
                back_to_backs_to_backs += 1
        return back_to_backs_to_backs

    def __eq__(self, other):
        return isinstance(other, Team) and self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(self.name)


# Class for representing a game between two teams
class Game:
    def __init__(self, home, away, date=-1):
        self.home = home
        self.away = away
        if date == -1:
            self.date = random.randint(1, DAYS)

    def __str__(self):
        return self.away.__str__() + " @ " + self.home.__str__()

    def __hash__(self):
        return hash((self.home, self.away, self.date))

    def __eq__(self, other):
        return isinstance(other, Game) and self.__hash__() == other.__hash__()


def calculate_distances(teams):
    for i, team1 in enumerate(teams):
        for j, team2 in enumerate(teams):
            if i < j:
                DISTANCES[(team1, team2)] = DISTANCES[(team2, team1)
                                                      ] = calculate_distance(team1, team2)


def calculate_distance(team1, team2):
    loc1 = (team1.loc.latitude, team1.loc.longitude)
    loc2 = (team2.loc.latitude, team2.loc.longitude)

    return distance.distance(loc1, loc2).km


def dist_teams(team1, team2):
    if team1 == team2:
        return 0
    if (team1, team2) in DISTANCES:
        return DISTANCES[(team1, team2)]
    else:
        raise ValueError("Distances have not been precomputed...")


def set_up_schedule():
    try:
        teams = pickle.load(open('teams.pkl', 'rb'))
    except FileNotFoundError:
        teams = create_teams()
        pickle.dump(teams, open('teams.pkl', 'wb'))

    games = add_games(teams)

    return teams, games


# Creating teams, querying location, hardcoded
def create_teams():
    teams = []

    teams.append(Team("Atlanta Hawks", get_first_city(
        'Atlanta'), 'Eastern', 'Southeast'))
    teams.append(Team("Boston Celtics", get_first_city(
        'Boston'), 'Eastern', 'Atlantic'))
    teams.append(Team("Brooklyn Nets", get_first_city(
        'Brooklyn'), 'Eastern', 'Atlantic'))
    teams.append(Team("Charlotte Hornets", get_first_city(
        'Charlotte, NC'), 'Eastern', 'Southeast'))
    teams.append(Team("Chicago Bulls", get_first_city(
        'Chicago'), 'Eastern', 'Central'))
    teams.append(Team("Cleveland Cavaliers", get_first_city(
        'Cleveland'), 'Eastern', 'Central'))
    teams.append(Team("Dallas Mavericks", get_first_city(
        'Dallas'), 'Western', 'Southwest'))
    teams.append(Team("Denver Nuggets", get_first_city(
        'Denver'), 'Western', 'Northwest'))
    teams.append(Team("Detroit Pistons", get_first_city(
        'Detroit'), 'Eastern', 'Central'))
    teams.append(Team("Golden State Warriors", get_first_city(
        'San Francisco'), 'Western', 'Pacific'))
    teams.append(Team("Houston Rockets", get_first_city(
        'Houston'), 'Western', 'Southwest'))
    teams.append(Team("Indiana Pacers", get_first_city(
        'Indianapolis'), 'Eastern', 'Central'))
    teams.append(Team("LA Clippers", get_first_city(
        'Los Angeles'), 'Western', 'Pacific'))
    teams.append(Team("LA Lakers", get_first_city(
        'Los Angeles'), 'Western', 'Pacific'))
    teams.append(Team("Memphis Grizzlies", get_first_city(
        'Memphis'), 'Western', 'Southwest'))
    teams.append(Team("Miami Heat", get_first_city(
        'Miami'), 'Eastern', 'Southeast'))
    teams.append(Team("Milwaukee Bucks", get_first_city(
        'Milwaukee'), 'Eastern', 'Central'))
    teams.append(Team("Minnesota Timberwolves", get_first_city(
        'Minneapolis'), 'Western', 'Northwest'))
    teams.append(Team("New Orleans Pelicans", get_first_city(
        'New Orleans'), 'Western', 'Southwest'))
    teams.append(Team("New York Knicks", get_first_city(
        'New York'), 'Eastern', 'Atlantic'))
    teams.append(Team("Oklahoma City Thunder", get_first_city(
        'Oklahoma City'), 'Western', 'Northwest'))
    teams.append(Team("Orlando Magic", get_first_city(
        'Orlando'), 'Eastern', 'Southeast'))
    teams.append(Team("Philadelphia 76ers", get_first_city(
        'Philadelphia'), 'Eastern', 'Atlantic'))
    teams.append(Team("Phoenix Suns", get_first_city(
        'Phoenix'), 'Western', 'Pacific'))
    teams.append(Team("Portland Trail Blazers", get_first_city(
        'Portland, OR'), 'Western', 'Northwest'))
    teams.append(Team("Sacramento Kings", get_first_city(
        'Sacramento, CA, USA'), 'Western', 'Pacific'))
    teams.append(Team("San Antonio Spurs", get_first_city(
        'San Antonio'), 'Western', 'Southwest'))
    teams.append(Team("Toronto Raptors", get_first_city(
        'Toronto'), 'Eastern', 'Atlantic'))
    teams.append(Team("Utah Jazz", get_first_city(
        'Salt Lake City'), 'Western', 'Northwest'))
    teams.append(Team("Washington Wizards", get_first_city(
        'Washington'), 'Eastern', 'Southeast'))

    return teams


def add_games(teams):
    not_play_4th_time = fill_not_play_4th_time(teams)

    games = set()
    for i, team1 in enumerate(teams):
        for j, team2 in enumerate(teams):
            if i < j:
                if team1.conf != team2.conf:
                    add_game(Game(team1, team2), games)
                    add_game(Game(team2, team1), games)

                elif team1.div == team2.div:
                    add_game(Game(team1, team2), games)
                    add_game(Game(team2, team1), games)
                    add_game(Game(team1, team2), games)
                    add_game(Game(team2, team1), games)
                else:
                    add_game(Game(team1, team2), games)
                    add_game(Game(team2, team1), games)
                    if team2 not in not_play_4th_time[team1] and team1 not in not_play_4th_time[team2]:
                        add_game(Game(team1, team2), games)
                        add_game(Game(team2, team1), games)
                    elif team2 in not_play_4th_time[team1]:
                        add_game(Game(team1, team2), games)
                    elif team1 in not_play_4th_time[team2]:
                        add_game(Game(team2, team1), games)
                    else:
                        raise ValueError("You should never get here")

    return games


def add_game(game, games):
    games.add(game)
    game.home.add_game(game)
    game.away.add_game(game)


def get_first_city(name):
    for i in range(1000):
        try:
            return geolocator.geocode(name, timeout=10)
        except geopy.exc.GeocoderTimedOut:
            print('GeoPy service timeout {}: trying again...'.format(i+1))
            pass
    raise geopy.exc.GeocoderTimedOut


def fill_not_play_4th_time(teams):
    div_opps = defaultdict(set)
    divs = defaultdict(list)
    for team1 in teams:
        divs[team1.div].append(team1)

    for i, team1 in enumerate(teams):
        for j, team2 in enumerate(teams):
            if i < j:
                if team1.conf == team2.conf and team1.div != team2.div:
                    div_opps[team1.div].add(team2.div)

    not_play_4th_time = defaultdict(list)

    for i, div in enumerate(divs):
        for j, div2 in enumerate(divs):
            if i < j and div in div_opps[div2]:
                for idx in range(5):
                    not_play_4th_time[divs[div][idx]].append(
                        divs[div2][idx % 5])
                    not_play_4th_time[divs[div2]
                                      [(idx+1) % 5]].append(divs[div][idx])

    return not_play_4th_time


##########################
### UTILS FOR PRINTING ###
##########################

def convert_day_to_date(days):
    start_date = datetime.strptime("10/16/18", "%m/%d/%y")
    return start_date + timedelta(days=days-1)


def save_schedule_to_textfile(teams, games):
    with open('schedule.txt', 'w') as f:
        for team in teams:
            print(team, file=f)
            for day in sorted(list(team.schedule.keys())):
                if len(team.schedule[day]) > 0:
                    print("{}: ".format(convert_day_to_date(
                        day).strftime('%Y-%m-%d')), end='', file=f)
                    for game in team.schedule[day]:
                        print(game, file=f)

            print("", file=f)

        print("Number of games: ", len(games), file=f)


############################################################
### DEFINING ACTIONS THAT CAN BE TAKEN DURING THE SEARCH ###
############################################################

class Action:
    def __init__(self, _type, game1=None, game2=None, date=None):
        self.type = _type
        self.game1 = game1

        if self.type == 'switch':
            self.game2 = game2
        elif self.type == 'move':
            self.new_date = date
            self.original_date = game1.date
        else:
            raise ValueError(
                "Action's type should be either 'switch' or 'move'.")

    def execute_action(self, games):
        if self.type == 'switch':
            games.remove(self.game1)
            games.remove(self.game2)
            self.game1.home.remove_game(self.game1)
            self.game2.home.remove_game(self.game2)
            self.game1.away.remove_game(self.game1)
            self.game2.away.remove_game(self.game2)

            self.game1.date, self.game2.date = self.game2.date, self.game1.date

            self.game1.home.add_game(self.game1)
            self.game2.home.add_game(self.game2)
            self.game1.away.add_game(self.game1)
            self.game2.away.add_game(self.game2)
            games.add(self.game1)
            games.add(self.game2)

        elif self.type == 'move':
            games.remove(self.game1)
            self.game1.home.remove_game(self.game1)
            self.game1.away.remove_game(self.game1)

            self.game1.date = self.new_date

            self.game1.home.add_game(self.game1)
            self.game1.away.add_game(self.game1)
            games.add(self.game1)

    def undo_action(self, games):
        if self.type == 'switch':
            games.remove(self.game1)
            games.remove(self.game2)
            self.game1.home.remove_game(self.game1)
            self.game2.home.remove_game(self.game2)
            self.game1.away.remove_game(self.game1)
            self.game2.away.remove_game(self.game2)

            self.game1.date, self.game2.date = self.game2.date, self.game1.date

            self.game1.home.add_game(self.game1)
            self.game2.home.add_game(self.game2)
            self.game1.away.add_game(self.game1)
            self.game2.away.add_game(self.game2)
            games.add(self.game1)
            games.add(self.game2)

        elif self.type == 'move':
            games.remove(self.game1)
            self.game1.home.remove_game(self.game1)
            self.game1.away.remove_game(self.game1)

            self.game1.date = self.original_date

            self.game1.home.add_game(self.game1)
            self.game1.away.add_game(self.game1)
            games.add(self.game1)

    def delta_loss(self, teams, games, original_loss):
        self.execute_action(games)
        new_loss = calc_loss(teams)
        self.undo_action(games)
        return new_loss - original_loss


def random_action(games):
    action = None
    while action is None:
        if random.random() < 0.5:
            # switch games
            game1 = random.sample(games, 1)[0]
            game2 = random.sample(games, 1)[0]

            if game1.date != game2.date:
                action = Action(_type='switch', game1=game1, game2=game2)
        else:
            game1 = random.sample(games, 1)[0]
            date = random.randint(1, DAYS)
            if game1.date != date:
                action = Action(_type='move', game1=game1, date=date)

    return action


################################################################################
### SIMULATED ANNEALING WITH EXPONENTIALLY DECREASING ACCEPTANCE PROBABILITY ###
################################################################################

ALPHA = 0.99
T_START = 1
T_FINAL = 0.00001
ACC_PROB_PARAM = 0.001


def calc_loss(teams):
    return sum([team.loss() for team in teams])


def acceptance_probability(delta_loss, T):
    if delta_loss < 0:
        return 1
    return math.exp(-ACC_PROB_PARAM * delta_loss / T)


def simulated_annealing(teams, games):
    current_loss = calc_loss(teams)
    T = T_START
    while T > T_FINAL:
        print(T, current_loss)

        i = 1
        while i <= 100:
            action = random_action(games)
            action_loss = action.delta_loss(teams, games, current_loss)
            ap = acceptance_probability(action_loss, T)
            if ap > random.random():
                action.execute_action(games)
                current_loss += action_loss
            i += 1
        T *= ALPHA


#####################
### MAIN FUNCTION ###
#####################

def main():
    teams, games = set_up_schedule()

    calculate_distances(teams)

    simulated_annealing(teams, games)

    save_schedule_to_textfile(teams, games)
    print(calc_loss(teams))
    for team in teams:
        print(team.name, ": Number of back-to-backs:", team.number_of_back_to_backs(),
              "Number of back-to-back-to-backs:", team.number_of_back_to_backs_to_backs())


if __name__ == '__main__':
    main()
