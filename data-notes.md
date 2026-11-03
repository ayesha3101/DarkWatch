# Data notes

Plain notes on the data we use, what we found in it, and what we decided.

## The short version

- **Global Fishing Watch (GFW)** tells us **where** fishing happens. We use it for the fishing-activity map.
- **Danish Maritime Authority (DMA)** AIS files tell us **how** real boats move. We use them to learn boat movement, to test the belief filter and to train the classifier.
- **The simulator** is our own made-up sea. The patrol demo (boats going dark, the patrol, fuel) runs there, because nobody publishes data on boats that hide on purpose.

## 1. Global Fishing Watch (fishing activity)

**What we asked for**
- Dataset: `public-global-fishing-effort:v5.0`
- Area: longitude 65.0 to 67.5, latitude 23.0 to 25.0 (off the Sindh coast)
- Dates: 2022-05-01 to 2022-12-01, as one total for the whole period
- Cell size: 0.1 degree

**What we got**
- One row for each map cell and flag (the country the boat is registered in).
- Each row has `lat`, `lon`, `hours`, `flag` and `vesselIDs`.
- `hours` is the number of hours boats looked like they were fishing.
- `vesselIDs` is only a count of boats, not a list.
- The `flag` is sometimes empty (boat country unknown).

**What we did not get**
- No boat positions or paths. Only totals for each cell.

**Things to remember**
- A cell can appear in several rows (one per flag), so we must add the hours up per cell.
- "Apparent fishing" means the boat moved like a fishing boat. It does not prove fishing, and it does not prove illegal fishing.
- We must give credit to Global Fishing Watch.

## 2. Danish Maritime Authority AIS (boat tracks)

**The file**
- `aisdk-2026-10-31.csv`, one full day. The file is very large, so we read only the first 2,000,000 rows. That covers about 00:00 to 02:37 on 31 October 2026.
- We kept only Class A and Class B boats: 1,794,652 rows from 2,764 boats.

**What is in it**
- Columns we use: Timestamp, Type of mobile, MMSI (boat ID), Latitude, Longitude, Navigational status, ROT (turn rate), SOG (speed), COG (course) and Ship type.
- Ship type counts: Cargo 536,149, Fishing 348,555, Undefined 299,835, Passenger 217,973, Tanker 199,073, and others.
- 400 boats with type Fishing have enough points. 1,789 other known boats do too.
- "Engaged in fishing" appears in 228,243 rows. The crew types this in, so we treat it as a hint only.
- Missing values: ROT 17.9%, COG 6.1%, SOG 0.3%.
- Speed: the middle value is 3.6 knots, the top is 49.2 knots, and a quarter of rows are 0.

**Terms:** we follow the DMA conditions for use of public data and mention them in the report.

### The small sample we saved (`data/processed/dma_sample.csv`)

We picked 80 boats (40 fishing, 40 other) so the two groups are equal. Real life is not equal, so we say that in the report. The sample has 24,030 rows and is 2.4 MB.

| Problem we found | How big | What we do about it |
|---|---|---|
| Invalid positions (latitude 91 means "no position") | 62 rows (0.3%), 38 boats | Remove these rows |
| Exact duplicate rows | 7,639 rows (about 32%) | Remove the copies |
| Boat not moving (speed 0) | 53.2% of rows | Decide in Week 2 whether to drop stopped points for the movement model |
| ROT missing | 29.9% | Work out turning from positions instead |
| COG missing | 13.1% | Work out direction from positions instead |
| Ship type changes inside one track | 49 of 80 boats | Ignore "Undefined" rows and use the one real type the boat reports |
| Two odd boat IDs starting with 97 (97000100 and 97000106) | 2 boats | Leave them out of the classifier data, because we do not know what they are |

After cleaning, all 80 boats are still there. A boat has 206 rows in the middle (shortest 34, longest 400), about 10 seconds apart. That is roughly 35 minutes of track.

**About the ship type check:** for every boat, the extra values were only "Undefined". No boat reports two different real types. So each boat has one clear label.

## 3. Decisions

1. **Labels:** a boat is "fishing" or "other" based on its real ship type, ignoring "Undefined" rows. We label per boat, never per row.
2. **Ship type is only the answer key.** The classifier must never use it as an input. It uses speed, turning and similar measures.
3. **Cleaning order (Week 2):** remove invalid positions, remove duplicates, remove the 97 boats, label per boat, drop very short tracks, resample to one point per minute, then work out speed and turning from positions.
4. **What is real and what is simulated:**
   - Real: the fishing-activity map, the movement model, the belief tests and the classifier.
   - Simulated: boats going dark, the patrol and its fuel.
5. **Region mismatch:** the Danish tracks are from Danish waters, but the fishing map is from the Sindh coast. We say this clearly in the report and we use the Danish tracks only to learn how boats move.
6. **Possible bias:** the sample only has boats whose type was announced in the rows we read. Boats that never announce a type are missing.

## 4. Grid format (shared by both of us)

- `cost[y, x]`: 1.0 means open water. A bigger number means harder to cross. `np.inf` means land. Every number that is not land is at least 1.
- `density[y, x]`: how much fishing happens in the cell, from 0 to 1.
- Row 0 is the top of the map (north), so y goes down as you move south.
- Run `python src/data.py` to see the simulator and its picture.

## 5. Still to do in Week 2

- Load the GFW file into Pandas, add up hours per cell and make the real `density` grid.
- Make the real `cost` grid (land and water).
- Write the cleaning script for the Danish tracks and save the clean version.
- Make 3 or 4 charts and write the findings in plain sentences.
