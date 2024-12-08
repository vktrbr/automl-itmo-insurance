# Dataset Reference

The target variable is distributed within the range of 25 to 4999. The 99th percentile value
consistently remains around 3850 over the months, suggesting that values exceeding 4000 can be
considered anomalous.

The mean value of the target variable is approximately 1100, while the median and standard deviation
are identical at 850.

Overall, the distribution of the variable remains stable over time. The only significant change was
observed in 2020, when the distribution shifted notably to the left, and the mean along with other
statistical measures decreased by 10.5%.

The distribution is skewed to the right, indicating a small number of policies with very high target
variable values. The mode is 25, reflecting a substantial number of policies (~60,000) with the
minimum payout. This value consistently remains the most common throughout the entire observation
period.

A significant proportion of payouts (38%) is concentrated in the range of 400 to 1000.

<img alt="Distribution Analysis" src="https://raw.githubusercontent.com/vktrbr/automl-itmo-insurance/main/reports/figures/target_analysis/png/PREMIUM_AMOUNT_DISTRIBUTION_analysis.png"/>
<img alt="Distribution Analysis" src="https://raw.githubusercontent.com/vktrbr/automl-itmo-insurance/main/reports/figures/target_analysis/png/PREMIUM_AMOUNT_DISTRIBUTION_BY_YEARS_analysis.png"/>
<img alt="Distribution Analysis" src="https://raw.githubusercontent.com/vktrbr/automl-itmo-insurance/main/reports/figures/target_analysis/png/PREMIUM_AMOUNT_QUANTILES_BY_MONTH_analysis.png"/>
<img alt="Distribution Analysis" src="https://raw.githubusercontent.com/vktrbr/automl-itmo-insurance/main/reports/figures/target_analysis/png/PREMIUM_AMOUNT_STATS_BY_MONTH_analysis.png"/>

---

---

::: dataset.get_dataset

---

::: dataset.preprocess_dataset
