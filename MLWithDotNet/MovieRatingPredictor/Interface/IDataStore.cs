using MovieRatingPredictor.Data;
using MovieRatingPredictor.Enums;
using Microsoft.ML;

namespace MovieRatingPredictor.Interface
{
    public interface IDataStore
    {
        IDataView GetData(MLContext mlContext,DataCategory dataCategory);
    }
}