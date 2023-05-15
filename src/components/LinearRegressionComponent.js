import React from 'react';
import { linearRegression } from 'simple-statistics';
import Shuffle from '../utils/Shuffle';
// Function to calculate Mean Absolute Percentage Error (MAPE)
const MeanAbsolutePercentageError = (actual, predicted) => {
  let total = 0;
  for (let i = 0; i < actual.length; i++) {
    total += Math.abs((actual[i] - predicted[i]) / actual[i]);
  }
  return (total / actual.length) * 100;
};

const LinearRegression = ({ data }) => {
  // Historical sales data
  const sales = data.sales;

  // Time in years, including the 5th month
  const time = [1, 2, 3, 4];

  // Combine sales and time arrays into a 2D array
  const salesTime = [];
  for (let i = 0; i < sales.length; i++) {
    salesTime.push([time[i], sales[i]]);
  }

  // Shuffle data array randomly
  const shuffledData = Shuffle(salesTime);

  // Split data into training and testing sets
  const splitIndex = Math.floor(shuffledData.length * 0.8); // 80% training, 20% testing
  const trainingData = shuffledData.slice(0, splitIndex);
  const testingData = shuffledData.slice(splitIndex);

  // Simple linear regression model using training data
  const { m, b } = linearRegression(trainingData);

  // Predict values for testing data
  const predictedValues = testingData.map((point) => m * point[0] + b);


  // Calculate Mean Absolute Percentage Error (MAPE)
  const mape = MeanAbsolutePercentageError(
    testingData.map((point) => point[1]),
    predictedValues
  );

  // Prediction for the 6th year using the entire data set
  const x = 6;
  const predictedSales = m * x + b;

  const accuracy = 100 - mape;

  return (
    <div>
      <h1>Modelo de Regresion Lineal Simple</h1>
      <p>Predicted sales for the current month: {predictedSales.toFixed(2)}</p>
      <p>Mean Absolute Percentage Error: {mape.toFixed(2)}%</p>
      <p>Accuracy: {accuracy.toFixed(2)}%</p>
    </div>
  );
};

export default LinearRegression;
