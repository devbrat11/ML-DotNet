using MovieRatingPredictor.Data;
using MovieRatingPredictor.Enums;
using MovieRatingPredictor.Model;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;

namespace MovieRatingPredictor.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class RatingController : ControllerBase
    {
        private static ITransformer _model;
        private static MLContext _context;

        [HttpGet]
        public IActionResult PredictRating()
        {
            var metric = _model.EvaluateModel(new CsvDataStore().GetData(_context, DataCategory.Test)).GetMetrics(_context);
            return Ok(metric);
        }

        [HttpPost]
        public IActionResult BuildModel()
        {
            _context = MlModelBuilder.CreateContext();
            var trainer = _context.Build().AddEstimator(_context.GetEstimator());
            _model = trainer.Train(new CsvDataStore().GetData(_context, DataCategory.Train));
            return Ok();
        }
    }
}