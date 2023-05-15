import './App.css';
import Clasificacion from './components/Clasificacion';
import MultipleLinearRegression from './components/MultipleLinearRegression';
import NeuronalNetowrking from './components/NeuronalNetworking';
import LinearRegression from './components/LinearRegressionComponent';
import { useState, useEffect } from 'react';
function App() {
  const [data, setData] = useState({});
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetch('http://35.231.78.51/fapi-dev/data.php/api', {
      mode: 'cors',
    })
      .then((response) => response.json())
      .then((data) => {
        setData({
          wastes: data.waste,
          sales: data.sales,
          demands: data.demands,
          offers: data.offers,
          products: data.productos,
          seassons: data.season,
          categories: data.categorias,
          subCategories: data.subCategorias,
          dailySales: data.ventas_x_dia,
        });
        setIsLoading(false);
      })
      .catch((err) => console.error(err));
  }, []);
  return (
    <div className="App">
      {isLoading ? (
        'is Loading...'
      ) : (
        <>
          <MultipleLinearRegression data={data} />
          <NeuronalNetowrking data={data} />
          <Clasificacion data={data} />
          <LinearRegression data={data} />
        </>
      )}
    </div>
  );
}

export default App;
