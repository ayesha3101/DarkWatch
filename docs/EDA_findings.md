# Exploratory Data Analysis: Findings (Lab 10)

For this lab, we worked with two real datasets: fishing activity data from Global Fishing Watch and vessel tracking data from the Danish Maritime Authority. We used Pandas to load, clean and analyse the data, then looked at the patterns and differences we could find.

The main work was done in `src/analysis.py`, `src/build_grids.py` and `src/clean_tracks.py`. We also created `src/verify_data.py` to check that the processed data was correct and ready to use. All 35 verification checks passed.

## 1. Fishing Activity (Global Fishing Watch)

We started with fishing activity data covering the sea off the Sindh coast, between 65.0° and 67.5° E longitude and 23.0° and 25.0° N latitude.

The dataset contained 374 rows. We checked for missing values and zero fishing hours and found that the data was already clean, so we did not need to remove any rows.

### What we found

- The 374 records covered 290 different map cells. Some cells appeared more than once because they had separate records for different flags, so we added their fishing hours together.
- The total recorded fishing activity was 16,579 hours.
- The median fishing hours per cell were 25.6, while the average was 57.2. This shows that a few cells had much higher fishing activity than most of the others.
- We identified 25 outlier cells (8.6% of the total) with more than 168.4 fishing hours. These cells accounted for 41.9% of all recorded fishing hours.
- The busiest 10% of cells accounted for 45.7% of the total activity.
- The highest fishing activity was concentrated around 24.5° N, 66.8° E, where three neighbouring cells recorded 537, 447 and 407 hours. We also found another busy area around 23.4°–23.7° N and 67.2°–67.3° E.
- Around 90% of the recorded hours were associated with the PAK flag, while the remaining 10% had no flag information.

### Grid generation and decisions

We converted the fishing activity data into a 26 × 21 grid. During this process, we checked the grid against the land and water areas. There were 45 land cells, and none of the fishing activity cells fell on land or outside the grid.

One thing we noticed was that fishing activity was not evenly distributed. A small number of cells had very high values compared to the rest. We decided to keep these outliers because they represent locations with genuinely high recorded activity, rather than obvious data errors.

However, using the raw values directly would make the hotspots dominate the map and make lower-activity areas difficult to see. To deal with this, we applied log scaling and then divided the values by the maximum. This made the differences easier to visualise without removing the important hotspots.

We also generated a separate land/water grid so that the demo could distinguish between land and sea.

## 2. Boat Tracks (Danish Maritime Authority)

For the second dataset, we used AIS vessel tracking data from the Danish Maritime Authority. We read the data in chunks rather than loading the entire file at once. For the analysis, we inspected the first 2,000,000 rows, covering approximately 00:00 to 02:37 on 31 October 2026.

We looked at the available columns, checked for missing or invalid values, and examined how different boats behaved. We also compared fishing boats with other types of vessels.

### Initial data inspection

Our first sample contained 80 boats and 24,030 rows. We found a few issues that needed to be handled before using the data:

- **Invalid positions:** 62 rows (around 0.3%) had invalid coordinates. A latitude value of 91, for example, was used to indicate that a position was unavailable.
- **Duplicate records:** 7,639 rows (around 32%) were exact duplicates.
- **Stationary boats:** Around 53% of the records had a speed of zero. Many of these boats appeared to be moored or not moving.
- **Missing movement information:** Rate of Turn (ROT) was missing in 29.9% of rows, while Course Over Ground (COG) was missing in 13.1%.
- **Changing ship types:** 49 out of 80 boats had more than one ship-type value in their records. However, the changes were only between Undefined and one actual ship type. No boat had two different real ship types.
- **Unknown boat IDs:** Two boat IDs started with 97. Since these IDs were not useful for identifying the vessel type in our analysis, we excluded them.

### Cleaning the data

We made a few decisions to make the tracks more consistent:

1. Removed records with invalid positions.
2. Removed exact duplicate records.
3. Excluded boat IDs starting with 97.
4. Assigned one ship-type label to each boat using its available real ship type, rather than treating every row as a separate label.
5. Kept one position per minute to make the tracks easier to compare.
6. Calculated movement information from the positions where possible, since ROT and COG were missing in some records.
7. Removed very short tracks and boats that were mostly stationary.

Our first cleaning attempt left us with only 22 boats. The main reason was that many of the boats in the original sample were not moving. We realised that selecting boats first and cleaning them afterwards was not giving us a useful sample for studying movement.

We changed the sampling process so that we cleaned the data first and then randomly selected moving boats using a fixed random seed. This made the results repeatable.

The new sample contained 120 boats and 64,154 rows. After resampling to one point per minute, we had 13,450 points. We then removed 19 boats that were mostly stationary. The final dataset contained **101 boats and 11,229 points**.

### What we found in the cleaned tracks

The final dataset contained:

- 43 fishing boats.
- 58 other boats.
- Around 111 points per boat on average, with a median of 101.
- The shortest track had 26 points, while the longest had 317.
- Most tracks covered roughly 1.5 to 2 hours.

We also compared the movement patterns of fishing boats and other vessels.

| Measurement | Fishing boats | Other boats |
|---|---:|---:|
| Median speed | 2.72 knots | 10.06 knots |
| Median turning angle per minute | 0.99° | 0.39° |

The difference was quite noticeable. Fishing boats generally moved more slowly and changed direction more often. Other vessels tended to move faster and follow straighter paths.

These differences could be useful when developing a model that looks for movement patterns resembling fishing activity.

### Decisions made during cleaning

There were a few decisions that were important for the later stages of the project:

- We assigned ship-type labels at the boat level instead of assigning them independently to every row. This avoids one boat having inconsistent labels across its track.
- Ship type is used only as the target label. We will not include it as an input feature, because that would give the classifier information it is supposed to predict.
- We will split the data by boat rather than randomly splitting individual rows. Otherwise, observations from the same boat could appear in both the training and testing sets, making the model's results look better than they really are.
- We intentionally selected moving boats because the purpose of this dataset is to study movement patterns. Stationary boats would not contribute much to that analysis.

One limitation of this approach is that the final sample does not represent the entire fleet. It focuses on moving boats with usable positions and ship-type information.

## 3. What This Means for Our Project

Both datasets are useful, but they serve different purposes.

The Global Fishing Watch data gives us a map of recorded fishing activity off the Sindh coast. We can use this to highlight areas with higher activity in our patrol demo. The density grid and land/water grid are already prepared for that purpose.

The Danish AIS dataset gives us individual vessel tracks that can help us understand how different types of boats move. The differences in speed and turning behaviour give us a starting point for building and testing a movement-based classifier.

The two datasets come from different regions, so we are not treating the Danish tracks as evidence of what is happening off the Sindh coast. We are using them to learn general vessel movement patterns and explore how those patterns might help identify activity that resembles fishing.

## 4. Limitations

There are a few limitations worth keeping in mind when interpreting these results.

**Fishing activity data only covers recorded AIS activity.** The Global Fishing Watch map shows vessels that were broadcasting AIS and were detected by the dataset. An empty cell does not necessarily mean that no fishing took place there. It could simply mean that no activity was recorded. This is especially important because vessels that do not broadcast AIS will not appear in the data.

**Apparent fishing is not proof of actual fishing.** A vessel's movement might resemble fishing behaviour, but that does not confirm that it was fishing. It also does not tell us whether an activity was legal or illegal. The model can identify patterns worth investigating, not make a final judgement.

**The AIS track sample is limited.** We analysed only a few hours from one day, and our final sample deliberately focuses on moving boats. We also excluded vessels without useful ship-type information. This means the results may not reflect how the full fleet behaves.

**The fishing map is relatively coarse.** Each grid cell is approximately 0.1 degrees wide, or around 11 km. This is useful for showing larger areas of activity, but it cannot pinpoint the exact location of an individual vessel.

## Conclusion

Overall, the EDA helped us understand what was available in both datasets and what needed to be done before using them in the project. The fishing activity data was already clean and could be converted into a density grid. The AIS data needed more work, especially around duplicates, invalid positions, missing values and stationary vessels.

We also found a useful difference between fishing boats and other vessels: in our cleaned sample, fishing boats moved more slowly and turned more frequently. This gives us something concrete to investigate in the next stage of the project.

The main takeaway is that both datasets are now prepared for their intended roles: one supports the geographic patrol demo, while the other provides vessel movement data for developing and testing the classifier.