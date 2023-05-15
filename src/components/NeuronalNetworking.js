import React, {useState, useEffect} from 'react';
import {Line} from 'react-chartjs-2';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

let accuracy = 0;
let error = 0;

export const NeuronalNetowrking = ({data}) => {
  console.log(data);
  const [jsonData, setJsonData] = useState(null);
  const [modeloVentas, setModeloVentas] = useState(null);
  const [chartData, setChartData] = useState(null);

  const loadData = async () => {
    setJsonData(data);
  };

  useEffect(() => {
    loadData();
  }, []);

  const trainModel = async () => {
    if (!jsonData) return;
    // Paso 1: Cargar y preparar los dato
    const fechas = jsonData.dailySales.map((dato) => {
      return new Date(dato.fecha).getTime();
    });

    const ventas = jsonData.dailySales.map((dato) => dato.total);
    const tensorFechas = tf.tensor2d(fechas, [fechas.length, 1]);
    const tensorVentas = tf.tensor2d(ventas, [ventas.length, 1]);

    // Normalizar los datos para que estén en el rango [0, 1]
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

    // Paso 2: Dividir los datos en conjuntos de entrenamiento y prueba
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

    // Paso 3: Crear el modelo de red neuronal
    const model = tf.sequential();
    model.add(
      tf.layers.dense({inputShape: [1], units: 16, activation: 'relu'})
    );
    model.add(tf.layers.dense({units: 1, activation: 'linear'}));

    // Paso 4: Compilar y entrenar el modelo
    model.compile({optimizer: 'adam', loss: 'meanSquaredError'});

    const history = await model.fit(tensorFechasTrain, tensorVentasTrain, {
      epochs: 100,
      validationData: [tensorFechasTest, tensorVentasTest],
      callbacks: tfvis.show.fitCallbacks(
        {name: 'Historial de entrenamiento'},
        ['loss', 'val_loss'],
        {height: 200, callbacks: ['onEpochEnd']}
      ),
    });

    tfvis.show.history({name: 'Historial de entrenamiento'}, history, [
      'loss',
      'val_loss',
    ]);
    // Calculate accuracy
    const evalOutput = model.evaluate(tensorFechasTest, tensorVentasTest, {
      verbose: 0,
    });
    let loss = evalOutput.dataSync();
    let acc = (1 - loss) * 100;
    accuracy = `${acc.toFixed(2)}%`;
    error = `${loss * 100}%`;
    setModeloVentas(model);
    generarPredicciones();
  };

  const generarPredicciones = async () => {
    if (!modeloVentas || !jsonData || !jsonData.length) return;

    const fechas = jsonData.map((dato) => new Date(dato.fecha).getTime());
    const tensorFechas = tf.tensor2d(fechas, [fechas.length, 1]);
    const [tensorFechasNorm] = tf.tidy(() => {
      const fechasMin = tensorFechas.min();
      const fechasMax = tensorFechas.max();

      const tensorFechasNorm = tensorFechas
        .sub(fechasMin)
        .div(fechasMax.sub(fechasMin));

      return [tensorFechasNorm];
    });

    const predicciones = modeloVentas.predict(tensorFechasNorm);
    const datosPredicciones = await predicciones.data();
    const ventasPredicciones = datosPredicciones.map((dato) => dato[0]);

    const nuevasFechas = [];
    const nuevasVentas = [];
    for (let i = 0; i < jsonData.length; i++) {
      nuevasFechas.push(new Date(jsonData[i].fecha));
      nuevasVentas.push(jsonData[i].cantidad);
      if (ventasPredicciones[i]) {
        nuevasFechas.push(new Date(jsonData[i].fecha));
        nuevasVentas.push(ventasPredicciones[i]);
      }
    }

    const data = {
      labels: nuevasFechas,
      datasets: [
        {
          label: 'Ventas',
          data: nuevasVentas,
          fill: false,
          backgroundColor: 'rgba(75,192,192,0.4)',
          borderColor: 'rgba(75,192,192,1)',
          pointBorderColor: 'rgba(75,192,192,1)',
          pointBackgroundColor: '#fff',
          pointBorderWidth: 1,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: 'rgba(75,192,192,1)',
          pointHoverBorderColor: 'rgba(220,220,220,1)',
          pointHoverBorderWidth: 2,
          pointRadius: 1,
          pointHitRadius: 10,
        },
      ],
    };

    const options = {
      scales: {
        xAxes: [
          {
            type: 'time',
            time: {
              unit: 'day',
              displayFormats: {
                day: 'DD MMM YY',
              },
            },
          },
        ],
        yAxes: [
          {
            ticks: {
              beginAtZero: true,
            },
            scaleLabel: {
              display: true,
              labelString: 'Ventas',
            },
          },
        ],
      },
    };

    setChartData(<Line data={data} options={options} />);
  };

  const handleModeloVentasClick = async () => {
    if (jsonData) {
      const model = await trainModel();
      setModeloVentas(model);
    }
  };

  return (
    <div>
      <h1> Modelo de Red Neuronal </h1>
      <p>Precision: {accuracy} </p>
      <p>Error: {error}</p>
      <button onClick={handleModeloVentasClick}>Modelo de Ventas</button>
      {chartData && <Line data={chartData} />}
    </div>
  );
};
