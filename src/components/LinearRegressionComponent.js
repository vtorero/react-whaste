import React from 'react';
import { linearRegression, linearRegressionLine } from 'simple-statistics';
import Shuffle from '../utils/Shuffle';
import MeanAbsolutePercentageError from '../utils/MeanAbsolutePercentageError';


const LinearRegression = ({ data }) => {
  const sales = data.sales;
  const time = [1, 2, 3, 4];
  const salesTime = [];
  for (let i = 0; i < sales.length; i++) {
    salesTime.push([time[i], sales[i]]);
  }
  const shuffledData = Shuffle(salesTime);

  const splitIndex = Math.floor(shuffledData.length * 0.8);
  const trainingData = shuffledData.slice(0, splitIndex);
  const testingData = shuffledData.slice(splitIndex);

  const lr = linearRegression(trainingData);
  const predict = linearRegressionLine(lr);
  const predictedValues = testingData.map((point) => predict(point[0]));

  const mape = MeanAbsolutePercentageError(
    testingData.map((point) => point[1]),
    predictedValues
  );

  const x = 1;
  const predictedSales = predict(x);

  const accuracy = 100 - mape;

  return (
    <div>
      <h1>Modelo de Regresión Lineal Simple</h1>
      <p>Predicted sales for the next month: {predictedSales.toFixed(2)}</p>
      <p>Mean Absolute Percentage Error: {mape.toFixed(2)}%</p>
      <p>Accuracy: {accuracy.toFixed(2)}%</p>
    </div>
  );
};

export default LinearRegression;
