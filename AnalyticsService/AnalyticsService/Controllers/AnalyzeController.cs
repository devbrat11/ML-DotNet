using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AnalyticsService.Data;
using AnalyticsService.Enums;
using AnalyticsService.Model;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace AnalyticsService.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class AnalyzeController : ControllerBase
    {
        [HttpGet]
        public IActionResult Analyze()
        {
            var context = MlModelBuilder.CreateContext();
            var trainer = context.Build().AddEstimator(context.GetEstimator());
            var model = trainer.Train(new CsvDataStore().GetData(context, DataCategory.Train));
            var metric = model.EvaluateModel(new CsvDataStore().GetData(context, DataCategory.Test)).GetMetrics(context);
            return Ok(metric);
        }
    }
}