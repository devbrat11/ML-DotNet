using System;
using System.IO;
using MovieRatingPredictor.Enums;
using MovieRatingPredictor.Interface;
using MovieRatingPredictor.Model;
using Microsoft.ML;

namespace MovieRatingPredictor.Data
{
    public class CsvDataStore:IDataStore
    {
        private readonly string _trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-train.csv");
        private readonly string _testingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "recommendation-ratings-test.csv");

        public IDataView GetData(MLContext mlContext, DataCategory dataCategory)
        {
            if (dataCategory == DataCategory.Train)
            {
                return mlContext.Data.LoadFromTextFile<MovieRating>(_trainingDataPath, hasHeader: true, separatorChar: ',');
            }
            else
            {
                return mlContext.Data.LoadFromTextFile<MovieRating>(_testingDataPath, hasHeader: true, separatorChar: ',');
            }
        }
    }
}
