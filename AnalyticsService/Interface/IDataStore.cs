using AnalyticsService.Data;
using AnalyticsService.Enums;
using Microsoft.ML;

namespace AnalyticsService.Interface
{
    public interface IDataStore
    {
        IDataView GetData(MLContext mlContext,DataCategory dataCategory);
    }
}