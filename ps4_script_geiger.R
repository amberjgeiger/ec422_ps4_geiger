library(tidyverse)
library(fpp3)
library(BVAR)

# ==========================================
# PART 2: A Business Forecasting Horse Race: ARIMA vs. ETS
# ==========================================

#2.1

# 2.1a

liquor_raw <- read_csv(
  "data/MRTSSM4453USN.csv",
  na = ".",
  col_types = cols(
    observation_date = col_date(format = "%Y-%m-%d"),
    MRTSSM4453USN = col_double()
  )
)

liquor <- liquor_raw |>
  transmute(Month = yearmonth(observation_date), sales = MRTSSM4453USN) |>
  as_tsibble(index = Month) |>
  filter(Month <= yearmonth("2019 Dec"))

has_gaps(liquor)


# 2.1b

liquor |>
  gg_season(log(sales)) +
  labs(
    title = "Seasonal Pattern of Log Liquor Store Sales",
    x = "Month",
    y = "Log(Sales)"
  )



#2.2

#2.2a

liquor_fit <- liquor |>
  model(arima = ARIMA(log(sales)),
        ets   = ETS(log(sales)))

liquor_fit |> select(arima) |> report()

liquor_fit |> select(ets) |> report()


# 2.2a Answer: For ARIMA, ARIMA(1,0,2)(2,1,1)[12] w/ drift 
# For ETS, ETS(A,A,A) 

#2.2b

#dof = (p + q + P + Q)
# p = 1, q = 2, P = 2, Q = 1
# dof = (1 + 2 + 2 + 1) = 6

liquor_fit |> select(arima) |> gg_tsresiduals()

liquor_fit |>
  select(arima) |>
  augment() |>
  features(.innov, ljung_box, lag = 24, dof = 6)


#2.3

#2.3a

liquor_cv <- liquor |>
  stretch_tsibble(.init = 120, .step = 1) |>
  filter(.id < max(.id))

liquor_cv_fit <- liquor_cv |>
  model(snaive = SNAIVE(sales),
        ets    = ETS(log(sales)),
        arima  = ARIMA(log(sales)))

saveRDS(liquor_cv_fit, "results/liquor_cv_fit.rds")

#2.3b

liquor_cv_fc <- liquor_cv_fit |>
  forecast(h = 1)

liquor_cv_fc |>
  accuracy(liquor, measures = list(RMSE = RMSE, MAE = MAE))


#2.4

#DM Test

liquor_errs <- liquor_cv_fc |>
  as_tibble() |>
  left_join(as_tibble(liquor) |> select(Month, actual = sales),
            by = "Month") |>
  mutate(e = actual - .mean) |>
  select(Month, .model, e) |>
  pivot_wider(names_from = .model, values_from = e)

forecast::dm.test(liquor_errs$arima, liquor_errs$ets, h = 1, power = 2)

#ARIMA VS SNAIVE

liquor_errs <- liquor_cv_fc |>
  as_tibble() |>
  left_join(as_tibble(liquor) |> select(Month, actual = sales),
            by = "Month") |>
  mutate(e = actual - .mean) |>
  select(Month, .model, e) |>
  pivot_wider(names_from = .model, values_from = e)

forecast::dm.test(liquor_errs$arima, liquor_errs$snaive,
                  h = 1, power = 2)


# ==========================================
# PART 3: Forecast Combination
# ==========================================

#3a

liquor_combo_fc <- liquor_cv_fit |>
  mutate(combo = (arima + ets) / 2) |>
  forecast(h = 1)

liquor_combo_fc |>
  accuracy(liquor, measures = list(RMSE = RMSE, MAE = MAE))

#3b

cor(liquor_errs$arima, liquor_errs$ets)