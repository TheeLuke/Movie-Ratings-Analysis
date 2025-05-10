import time
import argparse
import numpy as np
import pandas as pd

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, avg, mean, pow, sqrt, monotonically_increasing_id, lit, count, isnan, collect_list, expr, sum as spark_sum
    from pyspark.sql.types import IntegerType, FloatType, LongType, StructType, StructField, DoubleType, StringType, ArrayType
    from pyspark.storagelevel import StorageLevel
    from pyspark.ml.recommendation import ALS
    from pyspark.ml.evaluation import RankingEvaluator
    spark_available = True

    # Configuration
    APP_NAME = "Movie Recommendation ALS"
    RATINGS_SCHEMA = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("rating", FloatType(), True),
        StructField("timestamp", LongType(), True)
    ])
    TAGS_SCHEMA = StructType([
        StructField("userId", IntegerType(), True),
        StructField("movieId", IntegerType(), True),
        StructField("tag", StringType(), True),
        StructField("timestamp", LongType(), True)
    ])

    # Constants for Best Model (Determined from pre-executed Grid Search)
    BEST_RANK = 15
    BEST_REG_PARAM = 0.1

    # Interaction weights
    RATING_INTERACTION_VALUE = 1.0
    TAG_INTERACTION_VALUE = 0.5

except ImportError:
    print("PySpark is not installed or configured properly. Cannot run the script.")
    spark_available = False

# Spark helper functions
def create_spark_session(app_name):
    """Initializes and returns a SparkSession."""
    if not spark_available:
        print("Spark is not available. Cannot create session.")
        return None
    print("Initializing Spark Session...")
    try:
        spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .getOrCreate()
        spark.sparkContext.setLogLevel("WARN")
        print("Spark Session initialized successfully.")
        return spark
    except Exception as e:
        print(f"Error initializing Spark Session: {e}")
        raise

def load_and_prep_all_data(spark, ratings_hdfs_path, tags_hdfs_path):
    """Loads ratings and tags, applies schema, drops nulls, adds row ID, sorts."""
    if not spark_available or spark is None: return None, None
    print(f"Loading Ratings from {ratings_hdfs_path}...")
    try:
        ratings_df = spark.read.csv(ratings_hdfs_path, schema=RATINGS_SCHEMA, header=True)
        ratings_df = ratings_df.dropna(subset=["userId", "movieId", "rating", "timestamp"])
        print(f"Loaded {ratings_df.count()} valid ratings.")
    except Exception as e: print(f"Error loading ratings data: {e}"); raise

    print(f"Loading Tags from {tags_hdfs_path}...")
    try:
        tags_df = spark.read.csv(tags_hdfs_path, schema=TAGS_SCHEMA, header=True)
        tags_df = tags_df.dropna(subset=["userId", "movieId", "timestamp"])
        print(f"Loaded {tags_df.count()} valid tag applications.")
    except Exception as e: print(f"Error loading tags data: {e}"); raise

    print("Sorting ratings by timestamp and adding row ID...")
    # NO persist here - will be used once by split and then discarded
    ratings_sorted = ratings_df.sort("timestamp").withColumn("rowId", monotonically_increasing_id())

    print("Sorting tags by timestamp and adding row ID...")
    # NO persist here
    tags_sorted = tags_df.sort("timestamp").withColumn("rowId", monotonically_increasing_id())

    print("Data sorted and prepared for splitting.")
    return ratings_sorted, tags_sorted


def split_individual_data(df, df_name, persist_splits=False):
    if not spark_available or df is None:
         print(f"Skipping splitting for {df_name}: Spark or DataFrame not available.")
         return None, None, None

    print(f"Splitting {df_name} Data...")
    total_count = df.count()
    print(f"Total rows for splitting in {df_name}: {total_count}")

    if total_count == 0:
        print(f"Warning: {df_name} DataFrame is empty, returning empty splits.")
        empty_df = df.sparkSession.createDataFrame([], df.schema).drop("rowId")
        # Unpersist the input df if it was persisted for count/quantile
        if df.is_cached: df.unpersist()
        return empty_df, empty_df, empty_df

    train_prop=0.7; val_prop=0.15; relative_error = 0.01
    print(f"Calculating approximate quantiles for {df_name} rowId...")
    quantiles = df.stat.approxQuantile("rowId", [train_prop, train_prop + val_prop], relative_error)

    # Unpersist the input df after count/quantile
    if df.is_cached: df.unpersist()

    if len(quantiles) != 2: raise RuntimeError(f"approxQuantile failed for {df_name}. Got: {quantiles}")
    train_max_id, val_max_id = quantiles[0], quantiles[1]
    print(f"Splitting {df_name}: Train IDs <= {train_max_id}, Validation IDs ({train_max_id} < ID <= {val_max_id})")

    if "rowId" not in df.columns: raise ValueError("Column 'rowId' not found.")

    # Define splits using filters (lazy)
    train_df = df.filter(col("rowId") <= train_max_id).drop("rowId")
    validation_df = df.filter((col("rowId") > train_max_id) & (col("rowId") <= val_max_id)).drop("rowId")
    test_df = df.filter(col("rowId") > val_max_id).drop("rowId")

    # STRATEGIC CACHING
    # Only persist if requested
    if persist_splits:
        print(f"Persisting splits for {df_name}...")
        storage_level = StorageLevel.MEMORY_AND_DISK
        train_df.persist(storage_level)
        validation_df.persist(storage_level)
        test_df.persist(storage_level)

    # Trigger counts (actions) and print shapes
    print(f"{df_name} split definition complete. Calculating partition sizes...")
    start_count_time = time.time()
    train_count = train_df.count()
    val_count = validation_df.count()
    test_count = test_df.count()
    end_count_time = time.time()
    print(f"Partition size calculation took {end_count_time - start_count_time:.2f}s")

    print(f"  {df_name} Train:   {train_count} rows")
    print(f"  {df_name} Validation: {val_count} rows")
    print(f"  {df_name} Test:       {test_count} rows")

    return train_df, validation_df, test_df


def train_combined_implicit_als_model(ratings_train_df, tags_train_df, rank, regParam):
    """Combines rating/tag interactions and trains ALS with implicitPrefs=True."""
    if not spark_available or ratings_train_df is None or tags_train_df is None:
        print("Skipping ALS training: Spark or input DataFrames not available.")
        return None

    print(f"\n--- Training COMBINED Implicit ALS Model (rank={rank}, regParam={regParam}) ---")
    als_start_time = time.time()

    userCol = "userId"; itemCol = "movieId"; interactionCol = "combined_interaction"

    # Prepare interactions - these are lightweight transformations, no immediate need to persist
    ratings_interactions = ratings_train_df.select(userCol, itemCol, lit(RATING_INTERACTION_VALUE).alias("interaction_value"))
    tags_interactions = tags_train_df.select(userCol, itemCol, lit(TAG_INTERACTION_VALUE).alias("interaction_value"))

    print("Unioning ratings and tags training interactions...")
    combined_interactions = ratings_interactions.unionByName(tags_interactions)

    print("Aggregating interaction values per user-movie pair...")
    # Aggregation is expensive - CACHE the result before feeding to ALS
    aggregated_train_df = combined_interactions.groupBy(userCol, itemCol).agg(
        spark_sum("interaction_value").alias(interactionCol)
    )
    aggregated_train_df.persist(StorageLevel.MEMORY_AND_DISK) # Cache the crucial input to ALS
    agg_count = aggregated_train_df.count() # Action to materialize cache
    print(f"Created and cached aggregated training data with {agg_count} user-movie interaction scores.")
    if agg_count == 0:
        print("Aggregated training data is empty! Cannot train ALS.")
        if aggregated_train_df.is_cached: aggregated_train_df.unpersist()
        return None

    # ALS configuration
    als = ALS(maxIter=10, rank=rank, regParam=regParam, userCol=userCol,
              itemCol=itemCol, ratingCol=interactionCol, implicitPrefs=True, nonnegative=True)

    model = None
    try:
        print("Fitting Combined Implicit ALS model...")
        model = als.fit(aggregated_train_df)
        als_end_time = time.time()
        print(f"Finished training Combined Implicit ALS model. Time: {als_end_time - als_start_time:.2f}s")
    except Exception as e:
        print(f"ERROR during Combined Implicit ALS training: {e}")
        raise # Re-raise after potential cleanup
    finally:
        # Unpersist the aggregated training data once model is fit or if error occurs
        if aggregated_train_df.is_cached:
             aggregated_train_df.unpersist()
             print("Unpersisted aggregated training data.")

    return model


def evaluate_ranking_metrics(recommendations_df, eval_df, user_col="userId", item_col="movieId", rating_col="rating", k=100, rating_threshold=4.0):
    """Evaluates recommendations using ranking metrics."""
    if not spark_available or recommendations_df is None or eval_df is None:
         print(f"Skipping ranking evaluation: Spark or input DataFrames not available."); return None

    print(f"\n--- Evaluating Ranking Metrics (Top {k}, Rating Threshold >= {rating_threshold}) ---")
    eval_start_time = time.time()

    # Generate Ground Truth
    print(f"Generating ground truth (items rated >= {rating_threshold}, casting items to double)...")
    if rating_col not in eval_df.columns: print(f"Error: Rating column '{rating_col}' not found."); return None
    ground_truth_df = eval_df \
        .filter(col(rating_col) >= rating_threshold) \
        .groupBy(user_col) \
        .agg(collect_list(col(item_col).cast("double")).alias("actual_items"))
    # Check count *before* potentially expensive join
    gt_count = ground_truth_df.count()
    print(f"Generated ground truth for {gt_count} users with positive ratings.")
    if gt_count == 0: print("Warning: Ground truth empty after filter."); return None

    # The recommendations_df should already have 'userId' and 'recommendations_list'
    if 'userId' not in recommendations_df.columns or 'recommendations_list' not in recommendations_df.columns:
        print("Error: Input recommendations_df must contain 'userId' and 'recommendations_list' columns.")
        return None

    recs_count = recommendations_df.count()
    print(f"Received recommendations for {recs_count} users.")
    if recs_count == 0: print("Warning: No recommendations provided."); return None

    # Join ground truth with recommendations
    print("Joining ground truth with recommendations...")
    # Ensure recommendations_df only has userId and recommendations_list before join
    recs_for_join = recommendations_df.select("userId", "recommendations_list")
    eval_data_joined = ground_truth_df.join(recs_for_join, user_col, "inner")

    # Check count *before* evaluation
    eval_user_count = eval_data_joined.count()
    print(f"Joined ground truth and recommendations for {eval_user_count} users.")
    if eval_user_count == 0: print("Error: No common users between ground truth and recs for evaluation."); return None

    # Evaluate
    print("Calculating ranking metrics...")
    if "recommendations_list" not in eval_data_joined.columns or "actual_items" not in eval_data_joined.columns:
        print("Error: Required columns not found after join."); return None

    ranking_evaluator = RankingEvaluator(predictionCol="recommendations_list", labelCol="actual_items", k=k)
    metrics = {}
    metric_names = ["precisionAtK", "recallAtK", "ndcgAtK", "meanAveragePrecision"]
    for metric in metric_names:
        try:
            ranking_evaluator.setMetricName(metric)
            if "AtK" in metric: ranking_evaluator.setK(k)
            value = ranking_evaluator.evaluate(eval_data_joined)
            metrics[metric] = value if value is not None and not np.isnan(value) else 0.0
            metric_print_name = f"{metric}@{k}" if "AtK" in metric else metric
            print(f"  {metric_print_name}: {metrics[metric]:.4f}")
        except Exception as e: print(f"  Could not calculate {metric}: {e}"); metrics[metric] = 0.0

    eval_end_time = time.time()
    print(f"Ranking evaluation finished (took {eval_end_time - eval_start_time:.2f}s)")
    return metrics


def get_popularity_recommendations(spark, training_ratings_df, users_to_recommend_for_df, k=100):
    """
    Generates popularity-based recommendations.
    Recommends the top K most popular movies from the training data to every user in the evaluation set.
    Uses lit() followed by cast() for compatibility with older PySpark versions.
    """
    if not spark_available or spark is None or training_ratings_df is None or users_to_recommend_for_df is None:
         print("Skipping popularity recommendations: Spark or input DataFrames not available.")
         # Return a DataFrame with the expected schema but empty, to avoid downstream errors
         empty_schema = StructType([
             StructField("userId", IntegerType(), True),
             StructField("recommendations_list", ArrayType(DoubleType()), True)
         ])
         if spark is None:
             print("Cannot create dummy DataFrame without a Spark object.")
             return None
         return spark.createDataFrame([], empty_schema)


    print(f"\nGenerating Popularity-Based Recommendations (Top {k})...")
    popularity_start_time = time.time()

    # Calculate movie popularity based on rating counts in the training data
    print("Calculating movie popularity from training data...")
    movie_popularity = training_ratings_df.groupBy("movieId").count().orderBy(col("count").desc())

    # Get the top K most popular movieIds
    top_k_popular_movies = movie_popularity.limit(k).select("movieId").collect()
    top_k_movie_ids = [row['movieId'] for row in top_k_popular_movies]
    top_k_movie_ids_double = [float(movie_id) for movie_id in top_k_movie_ids]


    print(f"Identified top {len(top_k_movie_ids)} popular movies.")

    # Create a DataFrame with recommendations for each user in the evaluation set
    # Each user gets the same list of top K popular movies
    print(f"Generating recommendations for {users_to_recommend_for_df.count()} users...")

    # Get unique userIds from the dataframe we need to recommend for
    unique_users_df = users_to_recommend_for_df.select("userId").distinct()

    # Create a DataFrame where each row is a user and a list of the top K movie IDs
    popularity_recs_df = unique_users_df.withColumn(
        "recommendations_list",
        lit(top_k_movie_ids_double).cast(ArrayType(DoubleType()))
    )

    popularity_end_time = time.time()
    print(f"Finished generating Popularity Recommendations. Time: {popularity_end_time - popularity_start_time:.2f}s")

    # The output DataFrame should have columns: userId (int), recommendations_list (array<double>)
    return popularity_recs_df

def main(ratings_hdfs_path, tags_hdfs_path):
    """Main function using combined ratings/tags implicit ALS with optimized caching, and Popularity Baseline."""
    if not spark_available: return

    spark = None
    # Keep track only of DFs explicitly persisted that need cleanup
    persisted_dfs_registry = []
    try:
        spark = create_spark_session(APP_NAME)
        if spark is None: return

        # 1. Load and Prepare (NO persist here)
        ratings_sorted_df_with_id, tags_sorted_df_with_id = load_and_prep_all_data(spark, ratings_hdfs_path, tags_hdfs_path)
        if ratings_sorted_df_with_id is None:
            print("Failed to load ratings data. Exiting.")
            return


        # 2. Split Both Datasets Temporally
        # Persist ratings splits as val/test are needed later for evaluation
        ratings_train_df, ratings_val_df, ratings_test_df = split_individual_data(
            ratings_sorted_df_with_id, "Ratings", persist_splits=True
        )
        if ratings_train_df is None:
             print("Failed to split ratings data. Exiting.")
             return
        persisted_dfs_registry.extend([ratings_train_df, ratings_val_df, ratings_test_df])


        # Split tags, but only the train split is used directly, no need to persist splits
        tags_train_df, _, _ = split_individual_data(
            tags_sorted_df_with_id, "Tags", persist_splits=False
        )


        # Popularity Baseline
        print("\n--- Evaluating Popularity Baseline Model ---")
        # Generate popularity recommendations for users in the validation and test sets
        # need the unique users from the validation and test sets to generate recommendations for them
        val_users_df = ratings_val_df.select("userId").distinct() if ratings_val_df else None
        test_users_df = ratings_test_df.select("userId").distinct() if ratings_test_df else None

        if val_users_df is not None and ratings_train_df is not None:
             # Generate popularity recommendations for the validation users
             popularity_val_recs = get_popularity_recommendations(spark, ratings_train_df, val_users_df, k=100)

             # Evaluate the popularity baseline on the validation set
             print("\n--- Evaluating Popularity Baseline on Validation Set (Ranking Metrics) ---")
             # Passing recommendations_df directly
             # Ensure popularity_val_recs is not None in case get_popularity_recommendations failed
             if popularity_val_recs is not None:
                evaluate_ranking_metrics(popularity_val_recs, ratings_val_df, k=100)
             else:
                 print("Skipping Popularity Baseline evaluation on Validation Set: Failed to generate recommendations.")

        else:
             print("\nSkipping Popularity Baseline evaluation on Validation Set due to missing data.")


        if test_users_df is not None and ratings_train_df is not None:
            # Generate popularity recommendations for the test users
            popularity_test_recs = get_popularity_recommendations(spark, ratings_train_df, test_users_df, k=100)

            # Evaluate the popularity baseline on the test set
            print("\n--- Evaluating Popularity Baseline on Test Set (Ranking Metrics) ---")
            # Passing recommendations_df directly
            # Ensure popularity_test_recs is not None
            if popularity_test_recs is not None:
                evaluate_ranking_metrics(popularity_test_recs, ratings_test_df, k=100)
            else:
                 print("Skipping Popularity Baseline evaluation on Test Set: Failed to generate recommendations.")
        else:
             print("\nSkipping Popularity Baseline evaluation on Test Set due to missing data.")


        # Now that popularity evaluation is done, proceed with ALS training and evaluation

        # 3. Train COMBINED IMPLICIT ALS Model
        # This function handles caching/uncaching of the aggregated input internally
        # Only train ALS if both ratings_train_df and tags_train_df are available
        combined_implicit_model = None
        if ratings_train_df is not None and tags_train_df is not None:
            combined_implicit_model = train_combined_implicit_als_model(
                ratings_train_df, tags_train_df, rank=BEST_RANK, regParam=BEST_REG_PARAM
            )
            # Now that model is trained, we might unpersist the *inputs* to the training function
            if ratings_train_df and ratings_train_df.is_cached: ratings_train_df.unpersist(); persisted_dfs_registry.remove(ratings_train_df); print("Unpersisted ratings train split.")
            # tags_train_df was never persisted
        else:
            print("\nSkipping ALS model training due to missing training data (ratings or tags).")


        # 4. Evaluate COMBINED IMPLICIT Model using RANKING metrics
        if combined_implicit_model:
            print(f"\n--- Evaluating Combined Implicit Model (rank={BEST_RANK}, regParam={BEST_REG_PARAM}) ---")

            if ratings_val_df:
                print("\n--- Evaluating Combined Implicit Model on Validation Set (Ranking Metrics) ---")
                # The evaluate_ranking_metrics function is adapted to take recommendations_df directly.
                # Generate ALS recommendations for validation users.
                print("Generating ALS recommendations for validation users...")
                val_users_for_als_recs = ratings_val_df.select("userId").distinct()
                # Call recommendForUserSubset with the number of items (k) as a POSITIONAL argument
                als_val_recs = combined_implicit_model.recommendForUserSubset(val_users_for_als_recs, 100)

                # Transform ALS output to the expected schema: userId, recommendations_list: array<double>
                # Check if als_val_recs is not None before transforming
                if als_val_recs is not None:
                    als_val_recs_transformed = als_val_recs.withColumn(
                        "recommendations_list",
                         expr(f"transform(recommendations, rec -> CAST(rec.movieId AS DOUBLE))")
                    ).select("userId", "recommendations_list")
                    evaluate_ranking_metrics(als_val_recs_transformed, ratings_val_df, k=100)
                else:
                    print("Skipping ALS evaluation on Validation Set: Failed to generate recommendations.")
            else:
                 print("\nSkipping Combined Implicit Model evaluation on Validation Set due to missing data.")


            if ratings_test_df:
                print("\n--- Evaluating Combined Implicit Model on Test Set (Ranking Metrics) ---")
                # ratings_test_df should still be persisted
                print("Generating ALS recommendations for test users...")
                test_users_for_als_recs = ratings_test_df.select("userId").distinct()
                 # Call recommendForUserSubset with the number of items (k) as a POSITIONAL argument
                als_test_recs = combined_implicit_model.recommendForUserSubset(test_users_for_als_recs, 100)

                # Transform ALS output to the expected schema: userId, recommendations_list: array<double>
                # Check if als_test_recs is not None before transforming
                if als_test_recs is not None:
                    als_test_recs_transformed = als_test_recs.withColumn(
                        "recommendations_list",
                         expr(f"transform(recommendations, rec -> CAST(rec.movieId AS DOUBLE))")
                    ).select("userId", "recommendations_list")
                    evaluate_ranking_metrics(als_test_recs_transformed, ratings_test_df, k=100)
                else:
                    print("Skipping ALS evaluation on Test Set: Failed to generate recommendations.")

            else:
                print("\nSkipping Combined Implicit Model evaluation on Test Set due to missing data.")

        else:
            print("\nSkipping ALL ALS evaluations as the combined implicit model failed to train.")

    except Exception as e:
        print(f"\nAn error occurred in the main pipeline: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        if spark:
            print("\n--- Stopping Spark Session ---")
            print(f"Unpersisting remaining {len(persisted_dfs_registry)} cached DataFrames...")
            # Use the registry to unpersist only those we explicitly cached and tracked
            for i, df in enumerate(persisted_dfs_registry):
                try:
                    if hasattr(df, 'is_cached') and df.is_cached:
                        df.unpersist()
                except Exception as unpersist_e:
                    print(f"Warning: Error unpersisting DataFrame {i+1}: {unpersist_e}")
            spark.stop()
            print("Spark Session stopped.")

if __name__ == "__main__":
    if not spark_available:
        print("Exiting: PySpark is required but not available.")
    else:
        parser = argparse.ArgumentParser(description="Movie Recommendation Pipeline - Combined Implicit ALS + Ranking Eval v2")
        parser.add_argument("--ratings_path", type=str, required=True, help="HDFS path to ratings.csv")
        parser.add_argument("--tags_path", type=str, required=True, help="HDFS path to tags.csv")
        args = parser.parse_args()
        main(args.ratings_path, args.tags_path)