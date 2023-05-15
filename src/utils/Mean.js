function Mean(array) {
  const sum = array.reduce((sum, value) => sum + value, 0);
  return sum / array.length;
}

export default Mean;
