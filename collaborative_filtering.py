__author__ = 'vivek'
import codecs
import math
import time

training_set_path = "../netflix/TrainingRatings.txt"
test_set_path = "../netflix/TestingRatings.txt"

new_ratings = {}
reorg_ratings = []
movie_reviewers = {}
average_rating = {}

map_movieID = {}
map_userID = {}

def build_statistics():
    global training_set_path, new_ratings, reorg_ratings, movie_reviewers, map_movieID, map_userID

    ratings = {}

    custom_movieID = 0
    custom_userID = 0
    fh = codecs.open(training_set_path, "rU", "utf-8", errors="ignore")
    for line in fh:
        fields = line.strip().split(",")
        # 0 - movieID
        # 1 - userID
        # 2 - rating(integer) on scale of 5
        movieID = int(fields[0])
        userID = int(fields[1])
        rating = float(fields[2])

        if userID in ratings:
            ratings[userID][movieID] = rating
        else:
            ratings[userID] = {movieID: rating}
            map_userID[userID] = custom_userID
            custom_userID += 1

        if not(movieID in map_movieID):
            map_movieID[movieID] = custom_movieID
            custom_movieID += 1

    fh.close()

    print("#rows in users: %d" % len(ratings))
    print("#rows in movies: %d" % len(map_movieID))

    reorg_ratings = [[0 for i in range(len(map_movieID))] for j in range(len(ratings))]

    for user_id, reviews in ratings.items():
        new_user_id = map_userID[user_id]
        sum_rating = 0
        count = 0
        for movie_id, rating in reviews.items():
            new_movie_id = map_movieID[movie_id]
            reorg_ratings[new_user_id][new_movie_id] = rating

            if new_user_id in new_ratings:
                new_ratings[new_user_id][new_movie_id] = 0
            else:
                new_ratings[new_user_id] = {new_movie_id: 0}

            if not(new_movie_id in movie_reviewers):
                movie_reviewers[new_movie_id] = []

            movie_reviewers[new_movie_id].append(new_user_id)

            if rating > 0:
                sum_rating += rating
                count += 1
        if count != 0:
            average_rating[new_user_id] = sum_rating / count
        else:
            average_rating[new_user_id] = 0

def calc_correlation(userID, movies_considered, reviewer):
    global reorg_ratings, average_rating
    # return 1
    num = 0
    au_den = 0
    user_den = 0

    for movieID in movies_considered:
        if reorg_ratings[reviewer][movieID] > 0:
            num += (reorg_ratings[userID][movieID] - average_rating[userID]) * (reorg_ratings[reviewer][movieID] - average_rating[reviewer])
            au_den += pow((reorg_ratings[userID][movieID] - average_rating[userID]), 2)
            user_den += pow(reorg_ratings[reviewer][movieID] - average_rating[reviewer], 2)

    if au_den !=0 and user_den != 0:
        w = num / pow((au_den * user_den), 0.5)
    else:
        w = 0

    return w

def predict(user_id, movie_id):
    global new_ratings, reorg_ratings, movie_reviewers, average_rating, map_userID, map_movieID

    sum = 0
    k = 0

    userID = map_userID[user_id]
    movieID = map_movieID[movie_id]

    reviewers = movie_reviewers[movieID]
    movies_considered = new_ratings[userID].keys()

    for reviewer in reviewers:
        w = calc_correlation(userID, movies_considered, reviewer)

        if w > 0:
            k += 1
            r = reorg_ratings[reviewer][movieID] - average_rating[reviewer]
            sum += (w * r)

    if k != 0:
        delta_rating = sum / k
    else:
        delta_rating = 0

    prediction = average_rating[userID] + delta_rating

    return prediction

def report_accuracy():
    global test_set_path

    absolute_error_sum = 0
    squared_error_sum = 0
    count = 0

    fh = codecs.open(test_set_path, "rU", "utf-8", errors="ignore")
    for line in fh:
        fields = line.strip().split(",")
        userID = int(fields[1])
        movieID = int(fields[0])
        rating = float(fields[2])

        # print(count, "user_id:", userID, "movie_id:", movieID)

        predicted_rating = predict(userID, movieID)

        # print("rating: %f, predicted: %f" % (rating, predicted_rating))

        absolute_error_sum += math.fabs(predicted_rating - rating)
        squared_error_sum += math.pow((predicted_rating - rating), 2)
        count += 1

        if count > 1000:
            break

    if count != 0:
        mean_abs_error = absolute_error_sum / count
    else:
        mean_abs_error = 0

    if count != 0:
        rms_error = math.pow((squared_error_sum / count), 0.5)
    else:
        rms_error = 0

    return mean_abs_error, rms_error


print("Program execution started ...")

build_statistics()
mean_abs_error, rms_error = report_accuracy()
print("Mean absolute error: %f" % mean_abs_error)
print("Root Mean Square error: %f" % rms_error)

print("Program execution completed.")
