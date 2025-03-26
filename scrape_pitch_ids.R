library(baseballr)
library(tidyverse)

# 2024 MLB season dates
dates <- c(
  seq(from = as.Date("2022-04-07"), to = as.Date("2022-10-02"), by = "days"),
  seq(from = as.Date("2023-03-30"), to = as.Date("2023-10-01"), by = "days"),
  seq(from = as.Date("2024-03-28"), to = as.Date("2024-09-30"), by = "days")
)

gamepks <- map(dates, function(x) mlb_game_pks(date = x, level_ids = c(1))$game_pk, .progress = TRUE) %>% unlist()
gamepks <- unique(gamepks)

# select columns: 
colNames <- c("game_pk", "game_date", "playId", "isPitch", "home_team", "away_team",
              "matchup.pitcher.id", "matchup.pitcher.fullName", "matchup.batter.id", "matchup.batter.fullName",
              "about.inning", "about.halfInning", "atBatIndex", "pitchNumber",
              "count.balls.start", "count.strikes.start", "count.outs.start",
              "details.type.description")

pbp_w_ids <- map(gamepks, 
                 function(x) safely(
                   function(x) mlb_pbp(x) %>% 
                     select(all_of(colNames))
                 )(x)$result,
                 .progress = TRUE) %>%
  bind_rows() %>% 
  filter(isPitch)

write_csv(pbp_w_ids, "mlb_play_ids.csv")