API reference
=============

.. module:: qiskit_maestro

Everything below imports from ``qiskit_maestro``. The :doc:`guide` explains
the options in more detail.

Backend
-------

.. autoclass:: MaestroBackend
   :members: run, set_options, simulator_options, service

.. autoclass:: MaestroJob
   :members: result, status

Primitives
----------

.. autoclass:: MaestroSampler
   :members: run, from_backend, options, service

.. autoclass:: MaestroEstimator
   :members: run, run_evolution, from_backend, options, service

Qoro cloud jobs
---------------

With ``service=``, ``run()`` returns one of these as soon as the job is
submitted. The Qoro cloud section of the :doc:`guide` lists what runs there.

.. autoclass:: QoroCloudJob
   :members: result, status, cancel, job_id

.. autoclass:: QoroCloudPrimitiveJob
   :members: result, status, cancel, job_id

Exceptions
----------

.. autoexception:: MaestroError
.. autoexception:: UnsupportedInstructionError
.. autoexception:: UnsupportedNoiseError
.. autoexception:: MaestroGpuError
.. autoexception:: QoroCloudError
