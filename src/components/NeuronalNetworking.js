import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

// API endpoint
const API_URL = 'https://franz.kvconsult.com/fapi-dev/data.php/api';

const NeuronalNetworking = () => {
  const [accuracy, setAccuracy] = useState(0); // State for accuracy
  const [error, setError] = useState(0); // State for error (MSE)
  const [mae, setMae] = useState(0); // Mean Absolute Error (MAE)
  const [rmse, setRmse] = useState(0); // Root Mean Squared Error (RMSE)

  // Fetch data from the API
  const fetchData = async () => {
    try {
      const response = await fetch(API_URL);
      const data = await response.json();

      // Extract ventas_x_dia data
      const ventas_x_dia = data.ventas_x_dia;
      console.log("Sales data fetched:", ventas_x_dia);

      // Proceed to train the model with fetched data
      trainModel(ventas_x_dia);
    } catch (error) {
      console.error('Error fetching data from API:', error);
    }
  };

  // Train the neural network model
  const trainModel = async (ventasData) => {
    if (!ventasData || !ventasData.length) {
      console.error('No sales data available for training.');
      return;
    }

    // Prepare and normalize the data
    const fechas = ventasData.map((dato) => {
      // Split and process the 'fecha' field correctly
      const parts = dato.fecha.split('-').map(Number);
      const year = parts[0] < 50 ? 2000 + parts[0] : 1900 + parts[0];
      const time = new Date(year, parts[1] - 1, parts[2]).getTime(); // Correct date order
      return time;
    });

    const ventas = ventasData.map((dato) => dato.total);

    const tensorFechas = tf.tensor2d(fechas, [fechas.length, 1]);
    const tensorVentas = tf.tensor2d(ventas, [ventas.length, 1]);

    // Normalize the data
    const [tensorFechasNorm, tensorVentasNorm] = tf.tidy(() => {
      const fechasMin = tensorFechas.min();
      const fechasMax = tensorFechas.max();
      const ventasMin = tensorVentas.min();
      const ventasMax = tensorVentas.max();

      const tensorFechasNorm = tensorFechas
        .sub(fechasMin)
        .div(fechasMax.sub(fechasMin));
      const tensorVentasNorm = tensorVentas
        .sub(ventasMin)
        .div(ventasMax.sub(ventasMin));

      return [tensorFechasNorm, tensorVentasNorm];
    });

    // Train-test split
    const TRAIN_TEST_RATIO = 0.8;
    const [
      tensorFechasTrain,
      tensorVentasTrain,
      tensorFechasTest,
      tensorVentasTest,
    ] = tf.tidy(() => {
      const numExamples = tensorFechasNorm.shape[0];
      const numTrainExamples = Math.floor(numExamples * TRAIN_TEST_RATIO);
      const shuffledIndices = tf.util.createShuffledIndices(numExamples);
      const tensorShuffledIndices = tf.tensor1d(
        new Int32Array(shuffledIndices),
        'int32'
      );

      const tensorFechasTrain = tensorFechasNorm.gather(
        tensorShuffledIndices.slice(0, numTrainExamples)
      );
      const tensorVentasTrain = tensorVentasNorm.gather(
        tensorShuffledIndices.slice(0, numTrainExamples)
      );
      const tensorFechasTest = tensorFechasNorm.gather(
        tensorShuffledIndices.slice(numTrainExamples)
      );
      const tensorVentasTest = tensorVentasNorm.gather(
        tensorShuffledIndices.slice(numTrainExamples)
      );

      return [
        tensorFechasTrain,
        tensorVentasTrain,
        tensorFechasTest,
        tensorVentasTest,
      ];
    });

    // Build the model
    const model = tf.sequential();
    model.add(
      tf.layers.dense({ inputShape: [1], units: 16, activation: 'relu' })
    );
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    // Compile the model
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    // Train the model
    await model.fit(tensorFechasTrain, tensorVentasTrain, {
      epochs: 200,
      validationData: [tensorFechasTest, tensorVentasTest],
    });

    // Evaluate the model
    const evalOutput = model.evaluate(tensorFechasTest, tensorVentasTest, {
      verbose: 0,
    });
    let loss = evalOutput.dataSync()[0]; // Get the loss (MSE)
    let acc = (1 - loss) * 100;

    // Set accuracy and error (MSE)
    setAccuracy(`${acc.toFixed(2)}%`);
    setError(`${(loss * 100).toFixed(2)}%`);

    // Generate predictions and calculate additional metrics
    generarPredicciones(model, tensorFechasTest, tensorVentasTest);
  };

  // Generate predictions using the trained model and calculate additional metrics
  const generarPredicciones = async (model, tensorFechasTest, tensorVentasTest) => {
    if (!model) {
      console.error('No trained model available for predictions.');
      return;
    }

    const prediccionesTensor = model.predict(tensorFechasTest);
    const prediccionesData = await prediccionesTensor.data();
    const actualData = await tensorVentasTest.data();

    // Calculate MAE and RMSE
    const sumAbsoluteError = actualData.reduce((sum, actual, index) => sum + Math.abs(actual - prediccionesData[index]), 0);
    const mae = sumAbsoluteError / actualData.length;
    setMae(mae.toFixed(4));

    const sumSquaredError = actualData.reduce((sum, actual, index) => sum + Math.pow(actual - prediccionesData[index], 2), 0);
    const rmse = Math.sqrt(sumSquaredError / actualData.length);
    setRmse(rmse.toFixed(4));
  };

  // Fetch data when component mounts
  useEffect(() => {
    fetchData();
  }, []);

  return (
    <div>
      <h1>Modelo de Red Neuronal</h1>
      <p>Precisión: {accuracy}</p>
      <p>Error (MSE): {error}</p>
      <p>Error Absoluto Medio (MAE): {mae}</p>
      <p>Raíz del Error Cuadrático Medio (RMSE): {rmse}</p>
    </div>
  );
};

export default NeuronalNetworking;
