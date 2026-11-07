# Exploratory Data Analysis: Findings (Lab 10)

For this lab, we worked with two real datasets: fishing activity data from Global Fishing Watch and vessel tracking data from the Danish Maritime Authority. We used Pandas to load, clean and analyse the data, then looked at the patterns and differences we could find.

The main work was done in `src/analysis.py`, `src/build_grids.py` and `src/clean_tracks.py`. We also created `src/verify_data.py` to check that the processed data was correct and ready to use. All 35 verification checks passed.

## 1. Fishing Activity (Global Fishing Watch)

We started with fishing activity data covering the sea off the Sindh coast, between 65.0° and 67.5° E longitude and 23.0° and 25.0° N latitude.

The dataset contained 374 rows. We checked for missing values and zero fishing hours, and found that the data was already clean, so we did not need to remove any rows.

**What we found**

- The 374 records covered 290 different map cells. Some cells appeared more than once because they had separate records for different flags, so we added their fishing hours together.
- The total recorded fishing activity was 16,579 hours.
- The median fishing hours per cell were 25.6, while the average was 57.2. This shows that a few cells had much higher fishing activity than most of the others.
- We identified 25 outlier cells (8.6% of the total) with more than 168.4 fishing hours. Interestingly, these cells accounted for 41.9% of all recorded fishing hours.
- The busiest 10% of cells accounted for 45.7% of the total activity.
- The highest fishing activity was concentrated around 24.5° N, 66.8° E, where three neighbouring cells recorded 537, 447 and 407 hours. We also found another busy area around 23.4°–23.7° N and 67.2°–67.3° E.
- Around 90% of the recorded hours were associated with the PAK flag, while the remaining 10% had no flag information.
