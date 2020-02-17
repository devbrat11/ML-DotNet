using AnalyticsService.Data;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace AnalyticsService.Model
{
    public static class MlModelBuilder
    {
        public static MLContext CreateContext()
        {
            return new MLContext();
        }

        public static IEstimator<ITransformer> Build(this MLContext mlContext)
        {
            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: "movieId"));
            return estimator;
        }

        public static IEstimator<ITransformer> GetEstimator(this MLContext context)
        {
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "userIdEncoded",
                MatrixRowIndexColumnName = "movieIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            var estimator = context.Recommendation().Trainers.MatrixFactorization(options);
            return estimator;
        }

        public static IEstimator<ITransformer> AddEstimator(this IEstimator<ITransformer> baseEstimator, IEstimator<ITransformer> additionalEstimator)
        {
            return baseEstimator.Append(additionalEstimator);
        }

        public static ITransformer Train(this IEstimator<ITransformer> trainerEstimator, IDataView trainingDataView)
        {
            ITransformer model = trainerEstimator.Fit(trainingDataView);
            return model;
        }

        public static IDataView EvaluateModel(this ITransformer model, IDataView testDataView )
        {
            return model.Transform(testDataView);
        }

        public static EvaluationMetrics GetMetrics(this IDataView prediction,MLContext context)
        {
            var metrics = context.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");
            return new RegressionMetrics(metrics);
        }

        public static MovieRatingPrediction UseModelForSinglePrediction(this ITransformer model,MLContext context,MovieRating testInput)
        {
            var predictionEngine = context.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
            return predictionEngine.Predict(testInput);
        }
    }
}
