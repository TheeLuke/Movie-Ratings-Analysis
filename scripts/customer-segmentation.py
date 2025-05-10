import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
import time
import os
import random
import numpy as np
from scipy.stats import pearsonr

def find_movie_twins_minhash(ratings_df, num_perm=256, threshold=0.3, top_n=100, min_common_ratings_for_debug=10):
    print("\n--- Running MinHash LSH (Step 1 of Validation) ---")
    step_start_time = time.time()
    
    print(f"Original ratings count passed to find_movie_twins_minhash: {len(ratings_df)}")
    min_ratings_per_user = 20 #best threshold
    min_users_per_movie = 15 #best threshold

    print(f"Filtering users with < {min_ratings_per_user} ratings and movies with < {min_users_per_movie} user ratings.")
    
    user_rating_counts = ratings_df['userId'].value_counts()
    active_users = user_rating_counts[user_rating_counts >= min_ratings_per_user].index
    ratings_df = ratings_df[ratings_df['userId'].isin(active_users)]
    print(f"Ratings count after user filtering: {len(ratings_df)}")
    
    if not ratings_df.empty:
        movie_rating_counts = ratings_df['movieId'].value_counts()
        popular_movies = movie_rating_counts[movie_rating_counts >= min_users_per_movie].index
        ratings_df = ratings_df[ratings_df['movieId'].isin(popular_movies)]
        print(f"Ratings count after movie filtering: {len(ratings_df)}")
    else:
        print("DataFrame empty after user filtering.")

    if ratings_df.empty:
        print("Error: DataFrame is empty after all filtering. Adjust filter thresholds or check input data.")
        return None
        
    print("Preparing user movie sets...")
    RATING_THRESHOLD_FOR_LIKED = 3.0
    print(f"Defining user movie sets based on liked movies (rating >= {RATING_THRESHOLD_FOR_LIKED})...")
    
    liked_ratings_df = ratings_df[ratings_df['rating'] >= RATING_THRESHOLD_FOR_LIKED] 
    
    if liked_ratings_df.empty:
        print(f"Error: No movies found with rating >= {RATING_THRESHOLD_FOR_LIKED} after activity filtering. Adjust rating threshold or filters.")
        return None
        
    user_movie_sets = liked_ratings_df.groupby('userId')['movieId'].apply(set)
    
    user_movie_sets = user_movie_sets.apply(lambda s: {str(m) for m in s}) 
    if user_movie_sets.empty:
        print("Error: No user movie sets generated from 'liked' movies (post-filtering).")
        return None
    print(f"Found {len(user_movie_sets)} users with 'liked' movie sets (post-filtering).")

    print(f"Generating MinHash signatures (num_perm={num_perm})...")
    user_minhashes = {} 
    for user_id, movie_set in tqdm(user_movie_sets.items(), desc="Generating MinHashes"):
        m = MinHash(num_perm=num_perm)
        if not movie_set: continue
        for movie_id_str in movie_set: 
            m.update(movie_id_str.encode('utf8'))
        user_minhashes[user_id] = m
        
    if not user_minhashes:
         print("Error: No MinHash signatures generated from 'liked' movie sets.")
         return None

    print(f"Building LSH index (threshold={threshold}, num_perm={num_perm})...")
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    original_user_id_type = type(list(user_movie_sets.keys())[0]) if len(user_movie_sets) > 0 else int

    with lsh.insertion_session() as session:
         for user_id_orig, minhash_obj in tqdm(user_minhashes.items(), desc="Indexing MinHashes"):
             session.insert(str(user_id_orig), minhash_obj)

    print("Querying LSH index for candidate pairs...")
    candidate_pairs = set() 
    for user_id_orig_querier in tqdm(user_minhashes.keys(), desc="Querying LSH"):
        minhash_to_query = user_minhashes[user_id_orig_querier]
        result_neighbor_ids_str = lsh.query(minhash_to_query) 
        
        for neighbor_id_str in result_neighbor_ids_str:
            try:
                neighbor_id_orig = original_user_id_type(neighbor_id_str)
            except ValueError:
                continue
            
            if user_id_orig_querier == neighbor_id_orig: 
                continue
            
            pair = tuple(sorted((user_id_orig_querier, neighbor_id_orig)))
            candidate_pairs.add(pair)
            
    print(f"Found {len(candidate_pairs)} candidate pairs from LSH (based on 'liked' sets).")

    print("Calculating exact Jaccard similarity for LSH candidate pairs (based on 'liked' sets)...")
    results = []
    if candidate_pairs:
        ratings_df_for_debug_sets = liked_ratings_df.copy()
        ratings_df_for_debug_sets['movieId'] = ratings_df_for_debug_sets['movieId'].astype(str)


        for pair_count, pair in enumerate(tqdm(candidate_pairs, desc="Calculating Exact Similarities")):
            user_a_id, user_b_id = pair 
            set_a = user_movie_sets.get(user_a_id) # These are sets of 'liked' movies
            set_b = user_movie_sets.get(user_b_id)

            if set_a is None or set_b is None:
                continue
            
            intersection_size = len(set_a.intersection(set_b))
            union_size = len(set_a.union(set_b))
            similarity = 0.0 if union_size == 0 else intersection_size / union_size
            
            if 'print_debug_count_specific' not in locals(): print_debug_count_specific = 0
            if similarity > 0.5 and len(set_a) > 5 and print_debug_count_specific < 5 : 
                print_debug_count_specific +=1
                print(f"\nDEBUG: High Jaccard on LIKED sets: {similarity:.4f} (Set Size {len(set_a)}) pair: ({user_a_id}, {user_b_id})")

            if similarity >= threshold * 0.9:
                results.append((user_a_id, user_b_id, similarity))

    results.sort(key=lambda x: x[2], reverse=True)
    top_pairs_with_similarity = results[:top_n]
    
    print(f"MinHash LSH Step (Step 1) took {time.time() - step_start_time:.2f} seconds.")
    print("-" * 50)
    if not top_pairs_with_similarity:
         print(f"Warning: No top pairs found (from 'liked' sets) via MinHash LSH meeting criteria (Jaccard > ~{threshold * 0.9}, top {top_n}).")
         return [] 
    return top_pairs_with_similarity


def calculate_pearson_correlation_no_pivot(user1_id, user2_id, ratings_df, min_common_ratings=5):
    try:
        u1_ratings = ratings_df[ratings_df['userId'] == user1_id][['movieId', 'rating']]
        u2_ratings = ratings_df[ratings_df['userId'] == user2_id][['movieId', 'rating']]
        common_movies = pd.merge(u1_ratings, u2_ratings, on='movieId', suffixes=('_u1', '_u2'))
        if len(common_movies) < min_common_ratings:
            return np.nan
        if common_movies['rating_u1'].nunique() <= 1 or common_movies['rating_u2'].nunique() <= 1:
             return np.nan
        corr, _ = pearsonr(common_movies['rating_u1'], common_movies['rating_u2'])
        return corr
    except Exception as e:
        return np.nan

def validate_similarity_with_correlation(ratings_filepath, num_perm=256, threshold=0.3, top_n=100, num_random_pairs=100, min_common_ratings=5):
    overall_start_time = time.time()
    print(f"Loading data from {ratings_filepath}...")
    if not os.path.exists(ratings_filepath):
        print(f"Error: Ratings file not found at {ratings_filepath}")
        return None, None, None
    try:
        # Store the original df_ratings for correlation calculation, 
        # as find_movie_twins_minhash might filter its copy
        original_df_ratings = pd.read_csv(ratings_filepath)
        if not {'userId', 'movieId', 'rating'}.issubset(original_df_ratings.columns):
            print("Error: Ratings CSV must contain 'userId', 'movieId', and 'rating' columns.")
            return None, None, None
        # Pass a copy to find_movie_twins_minhash if it modifies it (it does with filtering)
        df_ratings_for_minhash = original_df_ratings.copy()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None, None, None

    top_pairs_with_similarity = find_movie_twins_minhash(df_ratings_for_minhash, num_perm, threshold, top_n)
    
    if top_pairs_with_similarity is None:
        print("Error occurred during MinHash LSH step. Cannot proceed.")
        return None, None, None 
    
    minhash_identified_pairs_ids = [(p[0], p[1]) for p in top_pairs_with_similarity]

    if not minhash_identified_pairs_ids:
         print("No MinHash pairs found to validate with correlation (after exact Jaccard).")
    
    print(f"\n--- Calculating Correlations (Step 2 of Validation) ---")
    correlation_step_start_time = time.time()
    print("Calculating correlations directly from original ratings data (no pivot table).")

    avg_corr_top_pairs = 0.0
    top_pairs_correlations_values = []
    if minhash_identified_pairs_ids:
        print(f"Calculating correlation for the top {len(minhash_identified_pairs_ids)} 'movie twin' pairs...")
        for pair_ids in tqdm(minhash_identified_pairs_ids, desc="Correlating Top MinHash Pairs"):
            corr = calculate_pearson_correlation_no_pivot(pair_ids[0], pair_ids[1], original_df_ratings, min_common_ratings)
            if not np.isnan(corr):
                top_pairs_correlations_values.append(corr)
        
        if top_pairs_correlations_values:
            avg_corr_top_pairs = np.mean(top_pairs_correlations_values)
        print(f"Average Pearson Correlation for MinHash 'twin' pairs: {avg_corr_top_pairs:.4f} (based on {len(top_pairs_correlations_values)} validly correlated pairs)")
    else:
        print("No MinHash 'twin' pairs to calculate correlation for.")

    all_user_ids = original_df_ratings['userId'].unique()
    avg_corr_random_pairs = 0.0
    if len(all_user_ids) < 2:
        print("Error: Not enough unique users in ratings to form random pairs.")
    else:
        print(f"Generating up to {num_random_pairs} random user pairs for baseline comparison...")
        random_pairs_set = set()
        minhash_pairs_frozen = {tuple(sorted(p)) for p in minhash_identified_pairs_ids}
        max_attempts_random = num_random_pairs * 20
        attempts = 0
        while len(random_pairs_set) < num_random_pairs and attempts < max_attempts_random:
            if len(all_user_ids) < 2: break 
            u1, u2 = random.sample(list(all_user_ids), 2)
            current_random_pair = tuple(sorted((u1, u2)))
            if current_random_pair not in minhash_pairs_frozen:
                random_pairs_set.add(current_random_pair)
            attempts += 1
        
        actual_random_pairs_generated = list(random_pairs_set)
        if len(actual_random_pairs_generated) < num_random_pairs:
            print(f"Warning: Could only generate {len(actual_random_pairs_generated)} unique random pairs.")

        random_pairs_correlations_values = []
        if actual_random_pairs_generated:
            print(f"Calculating correlation for {len(actual_random_pairs_generated)} random pairs...")
            for pair in tqdm(actual_random_pairs_generated, desc="Correlating Random Pairs"):
                corr = calculate_pearson_correlation_no_pivot(pair[0], pair[1], original_df_ratings, min_common_ratings)
                if not np.isnan(corr):
                    random_pairs_correlations_values.append(corr)
            if random_pairs_correlations_values:
                avg_corr_random_pairs = np.mean(random_pairs_correlations_values)
            print(f"Average Pearson Correlation for random pairs: {avg_corr_random_pairs:.4f} (based on {len(random_pairs_correlations_values)} validly correlated pairs)")
        else:
            print("No random pairs generated or correlated.")
            
    print(f"Correlation calculations (Step 2) took {time.time() - correlation_step_start_time:.2f} seconds.")
    print(f"\nValidation process total time: {time.time() - overall_start_time:.2f} seconds.")
    return avg_corr_top_pairs, avg_corr_random_pairs, top_pairs_with_similarity


##########################main###########################
if __name__ == "__main__":

    RATINGS_FILE = 'ratings.csv' 

    NUM_PERMUTATIONS = 256 
    LSH_THRESHOLD = 0.3
    TOP_N_PAIRS_TO_DISPLAY = 100 
    NUM_RANDOM_PAIRS_FOR_BASELINE = 100
    MIN_COMMON_RATINGS_FOR_CORR = 15

    validation_results_tuple = validate_similarity_with_correlation(
        ratings_filepath=RATINGS_FILE,
        num_perm=NUM_PERMUTATIONS,
        threshold=LSH_THRESHOLD,
        top_n=TOP_N_PAIRS_TO_DISPLAY,
        num_random_pairs=NUM_RANDOM_PAIRS_FOR_BASELINE,
        min_common_ratings=MIN_COMMON_RATINGS_FOR_CORR
    )

    if validation_results_tuple is not None and \
       validation_results_tuple[0] is not None and \
       validation_results_tuple[1] is not None and \
       validation_results_tuple[2] is not None:
        
        avg_corr_twins, avg_corr_random, top_minhash_pairs_with_jaccard = validation_results_tuple
        num_actual_twin_pairs_found_by_minhash = len(top_minhash_pairs_with_jaccard)

        print("\n--- Final Validation Results ---")
        print(f"LSH Parameters Used: Num Permutations={NUM_PERMUTATIONS}, Jaccard Threshold={LSH_THRESHOLD}")
        print(f"Min Common Ratings for Correlation: {MIN_COMMON_RATINGS_FOR_CORR}")
        print(f"Filtering: Users with < {20} ratings, Movies with < {15} ratings (adjust values in find_movie_twins_minhash)") # Manual update for print
        print(f"Number of 'Movie Twin' pairs found by MinHash & Jaccard (up to top {TOP_N_PAIRS_TO_DISPLAY}): {num_actual_twin_pairs_found_by_minhash}")
        print(f"Average Pearson Correlation for these 'Movie Twin' Pairs: {avg_corr_twins:.4f}")
        print(f"Average Pearson Correlation for Randomly Selected Pairs: {avg_corr_random:.4f}")

        if top_minhash_pairs_with_jaccard:
            print(f"\n--- Top {num_actual_twin_pairs_found_by_minhash} 'Movie Twin' Pairs (UserID1, UserID2, Jaccard Similarity) ---")
            for i, pair_data in enumerate(top_minhash_pairs_with_jaccard):
                print(f"{i+1}. User IDs: ({pair_data[0]}, {pair_data[1]}), Jaccard Similarity: {pair_data[2]:.4f}")
        elif num_actual_twin_pairs_found_by_minhash == 0 :
            print(f"\nNo 'Movie Twin' pairs were found meeting the LSH/Jaccard criteria to display.")
        
        print("\n--- Conclusion ---")
        if num_actual_twin_pairs_found_by_minhash > 0 and avg_corr_twins > avg_corr_random:
            print("The 'movie twin' pairs (found by MinHash LSH) have a meaningfully higher average rating correlation")
            print("than randomly selected pairs. This suggests the Jaccard similarity on user movie sets captured taste alignment.")
        elif num_actual_twin_pairs_found_by_minhash > 0 and avg_corr_twins < avg_corr_random:
             print("Unexpectedly, the random pairs have a higher average rating correlation than the MinHash pairs.")
             print("This could indicate: data sparsity, LSH parameters needing tuning, filtering choices, or that Jaccard similarity on movie sets")
             print("was not a strong indicator of correlated rating behavior for this dataset/configuration.")
        elif num_actual_twin_pairs_found_by_minhash == 0 :
             print("No 'movie twin' pairs were found by MinHash LSH. Cannot compare correlation.")
             print("Consider lowering LSH_THRESHOLD, adjusting filtering, increasing data, or checking data processing steps.")
        else: 
             print("The average rating correlation is similar between 'movie twin' pairs and random pairs,")
             print("or there were issues in generating one of the sets for comparison.")
    else:
        print("\nValidation process could not be completed due to critical errors in data loading or initial MinHash LSH steps.")
        print("Please review error messages above to diagnose the issue (e.g., file not found, CSV format, empty dataset after filtering).")