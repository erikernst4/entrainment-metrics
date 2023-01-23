Continuous TimeSeries
=====================

Using Continuous TimeSeries
---------------------------

When you already have a list of InterpausalUnit with some feature of interest calculated you can already create a TimeSeries object for some feature that the IPUs have precalculated. Here's an example using knn.

.. code-block:: python

   from entrainment_metrics.continuous import TimeSeries

   time_series: TimeSeries = TimeSeries(
       interpausal_units=ipus,
       feature="FEATURE_CALCULATED",
       method='knn',
       k=8,
   )

Moreover, you can predict over some points in time (a float or an np.ndarray) or an interval between some start and end with some granularity.

.. code-block:: python

   import numpy as np

   time_series_values: np.ndarray = time_series.predict_interval(
       start=0.5,
       end=44.4,
       granularity=0.0001,
   )

   another_time_series_values: np.ndarray = time_series.predict(
       np.arange(10, 20, 0.02)
   )


Calculating Metrics
-------------------

When you already have some TimeSeries is when fun starts. Now you are able to calculate some metrics over them. The metrics available are ‘proximity’, ‘convergence’, and ‘synchrony’. Calculating any of these metrics is as easy as running the following code:

.. code-block:: python

   from entrainment_metrics.continuous import calculate_metric
   metric_result: float = calculate_metric(
       "synchrony",
       time_series_a,
       time_series_b,
   )
