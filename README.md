# DSAI HW2 - AutoTradingru,

## 方法
使用LSTM模型，用今天的開盤價去預測明天的開盤價，只要預測結果比今天的高且手中握有股份就賣出，相反的只要預測結果比較低，就買進，都無法操作的話就不動作

## 執行
python trader.py --training training_data.csv --testing testing_data.csv --output output.csv
