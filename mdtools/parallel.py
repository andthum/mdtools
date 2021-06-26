# This file is part of MDTools.
# Copyright (C) 2021, The MDTools Development Team and all contributors
# listed in the file AUTHORS.rst
#
# MDTools is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# MDTools is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License
# along with MDTools.  If not, see <http://www.gnu.org/licenses/>.


"""Functions and clases for parallel computing."""


# Standard libraries
import os
import warnings
import traceback
import atexit
import uuid
import pickle
import multiprocessing as mp
# Local application/library specific imports
import mdtools as mdt


class ProcessPool:
    """
    Create a pool of worker processes which will carry out tasks
    submitted to the pool.
    
    This is a very basic custom version of Python's built-in
    :class:`multiprocessing.pool.Pool` class.  The problem with the
    built-in :class:`multiprocessing.pool.Pool` is that the message pipe
    between the parent process and its child processes has a limited
    size.  The pipe may get stuffed when you send too much data between
    the processes resulting in an error or, in the worst case,
    in a dead lock (see e.g. https://bugs.python.org/issue8426).  This
    custom version of a process pool circumvents this issue by pickling
    the data to a file and only sending the filename through the pipe.
    At the other end of the pipe, the data are unpickled from the file
    and can be processed.
    
    The file names follow a specific pattern.  Files sheduling tasks to
    woker (i.e. child) processes follow the pattern
    :file:`.pid<ProcessID>_pool<PoolNumber>_task<TaskNumber>_args_uuid_<UUID>.pkl`
    if they contain non-keyword arguments (:file:`args`) or
    :file:`.pid<ProcessID>_pool<PoolNumber>_task<TaskNumber>_kwargs_uuid_<UUID>.pkl`
    if they contain keyword arguments (:file:`kwargs`).  Files which are
    sent back from the worker processes to the parent process follow the
    pattern
    :file:`.pid<ProcessID>_pool<PoolNumber>_task<TaskNumber>_result_uuid_<UUID>.pkl`.
    Files are deleted automatically after use.  You might only have to
    clean up your working directory if the program crashes during
    execution.  Note that the leading dot :file:`.` marks the files
    hidden on Unix systems.  To show them, you have to invoke
    :command:`ls -a`.
    
    This process pool was only implemented to circumvent the above
    mentioned issue and has therefore only a very basic functionality,
    not comparable with the sophisticated functionality of the built-in
    :class:`multiprocessing.pool.Pool`.
    """
    
    #: Counts how many instances of :class:`ProcessPool`` have been
    #: created.
    _counter = 0
    
    def __init__(self, nprocs=None, parse_file=True):
        """
        Initialize a pool of worker processes.
        
        Parameters
        ----------
        nprocs : int
            The number of worker processes to use, i.e. the number of
            child processes to spawn.  If ``None`` (default), the number
            worker processes is inferred from
            :func:`mdtools.run_time_info.get_num_CPUs`.
        parse_file : bool, optional
            If ``True`` (default), send data between processes in files
            as described above instead of piping them.  This option sets
            the global behaviour of the pool.  It can also be set
            indivudially for each task submitted to the pool (see
            :meth:`submit_task`).
        """
        self._poolnum = ProcessPool._counter
        ProcessPool._counter += 1
        atexit.register(self.__del__)
        if nprocs is None:
            self._nprocs = mdt.rti.get_num_CPUs()
        else:
            self._nprocs = nprocs
        if self._nprocs < 1:
            raise ValueError("The number of processes ({}) must be"
                             " positive".format(self._nprocs))
        elif self._nprocs == 1:
            warnings.warn("The number of processes is only one. A serial"
                          " code is likely to be faster due to the"
                          " overhead of multiprocessing", RuntimeWarning)
        elif self._nprocs > mdt.rti.get_num_CPUs():
            warnings.warn("The number of processes ({}) is larger than"
                          " the number of available CPUs ({}). This will"
                          " probably lead to performance degradation"
                          .format(self._nprocs, mdt.rti.get_num_CPUs()),
                          RuntimeWarning)
        self._parse_file = bool(parse_file)
        self._pool_closed = False
        self._tasknum = 0
        self._ntasks_done = 0
        self._taskq = mp.Queue()
        self._resultq = mp.Queue()
        self._sentinel = None
        self._procs = []
        for i in range(self._nprocs):
            proc = mp.Process(target=self._worker,
                              args=(self._taskq,
                                    self._resultq,
                                    self._sentinel))
            proc.start()
            atexit.register(proc.terminate)
            self._procs.append(proc)
    
    def __del__(self):
        """Terminate the process pool."""
        self.terminate()
    
    def n_tasks_submitted(self):
        """
        Get the total number of tasks submitted to the pool.
        
        Returns
        -------
        _tasknum : int
            The *total* number of tasks submitted to the pool (including
            already finished tasks).
        """
        return self._tasknum
    
    def n_tasks_done(self):
        """
        Get the total number of done tasks.
        
        Returns
        -------
        _ntasks_done : int
            The total number of already finished tasks.
        """
        return self._ntasks_done
    
    def _worker(self, inq, outq, sentinel):
        """
        A wrapper function executed by the worker processes.
        
        Each spawned worker (child) process runs one :meth:`_worker`
        method.  The :meth:`_worker` method reads a task from `inq` (the
        task queue of the parent process), executes the task and send
        its result back to the parent process via `outq` (the result
        queue of the parent process).  If a worker reads `sentinel` from
        `inq`, it stops and the process running the :meth:`_worker`
        method terminates.
        
        Parameters
        ----------
        inq : multiprocessing.Queue
            Input queue from wich to read tasks.
        outq : multiprocessing.Queue
            Output queue in which to put the results.
        sentinel : object
            The sentinel object indicating the end of the input queue.
        
        Returns
        -------
        tasknums : list
            List of task numbers of all tasks processed by the worker
            process running this specific :meth:`_worker` method in the
            order of processing.
        """
        try:
            tasknums = []
            for tasknum, parse_file, func, args, kwargs in iter(inq.get,
                                                                sentinel):
                if parse_file:
                    with open(args, 'rb') as f:
                        arguments = pickle.load(f)
                    os.remove(args)
                    args = arguments
                    with open(kwargs, 'rb') as f:
                        keyword_arguments = pickle.load(f)
                    os.remove(kwargs)
                    kwargs = keyword_arguments
                result = func(*args, **kwargs)
                if parse_file:
                    fresult = (".pid" + str(os.getpid()) +
                               "_pool" + str(self._poolnum) +
                               "_task" + str(tasknum) +
                               "_result"
                               "_uuid_" + str(uuid.uuid4()) +
                               ".pkl")
                    with open(fresult, 'wb') as f:
                        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
                    result = fresult
                outq.put((tasknum, parse_file, result))
                tasknums.append(tasknum)
            return tasknums
        except Exception as e:
            print("An exception was raised in {} (PID: {}) while"
                  " processing task {} of ProcessPool {}:"
                  .format(mp.current_process().name,
                          os.getpid(),
                          tasknum,
                          self._poolnum))
            traceback.print_exc()
            outq.put((tasknum, e))
    
    def submit_task(self, func, args=(), kwargs={}, parse_file=None):
        """
        Submit a task to the process pool.  The task will start as soon
        as a free worker (child) process is available.
        
        Parameters
        ----------
        func : function
            The function to execute.
        args : tuple
            Non-keyword arguments to parse to `func`.
        kwargs : dict
            Keyword arguments to parse to `func`.
        parse_file : bool
            If ``True``, send data between processes in files as
            described above.  The default (``None``) is to infer the
            behaviour from the `parse_file` argument that was used for
            the construction of this :class:`ProcessPool`.
        
        Returns
        -------
        _tasknum : int
            The number assigned to the submitted task.  Tasks are
            numberd sequentially according to their submission time (the
            first submitted task gets number 0).
        """
        if parse_file is None:
            parse_file = self._parse_file
        if parse_file:
            fargs = (".pid" + str(os.getpid()) +
                     "_pool" + str(self._poolnum) +
                     "_task" + str(self._tasknum) +
                     "_args" +
                     "_uuid_" + str(uuid.uuid4()) +
                     ".pkl")
            with open(fargs, 'wb') as f:
                pickle.dump(args, f, pickle.HIGHEST_PROTOCOL)
            args = fargs
            fkwargs = (".pid" + str(os.getpid()) +
                       "_pool" + str(self._poolnum) +
                       "_task" + str(self._tasknum) +
                       "_kwargs" +
                       "_uuid_" + str(uuid.uuid4()) +
                       ".pkl")
            with open(fkwargs, 'wb') as f:
                pickle.dump(kwargs, f, pickle.HIGHEST_PROTOCOL)
            kwargs = fkwargs
        self._taskq.put((self._tasknum, parse_file, func, args, kwargs))
        self._tasknum += 1
        return self._tasknum - 1
    
    def get_results(self):
        """
        Get the results of all tasks that have not been collected, yet.
        
        This method blocks until all undone tasks are done and their
        results are fetched.  Results are returned in the order of task
        submission (FIFO: first in, first out).
        
        Returns
        -------
        results : tuple
            A tuple of all collected results.
        """
        results = []
        for i in range(self._ntasks_done, self._tasknum):
            result = self._resultq.get()
            if isinstance(result[-1], BaseException):
                print("Task {} exited with error:"
                      .format(result[0]))
                raise result[-1]
            else:
                results.append(result)
            self._ntasks_done += 1
        results.sort()
        tasknum, parse_file, results = tuple(zip(*results))
        results = list(results)
        for i, fresult in enumerate(results):
            if parse_file[i]:
                with open(fresult, 'rb') as f:
                    results[i] = pickle.load(f)
                os.remove(fresult)
        return tuple(results)
    
    def close(self):
        """
        Close the process pool.
        
        Prevents any more tasks from being submitted to the pool.  Once
        all the tasks have been completed, the worker processes will
        exit.  You should collect your result with :meth:`get_results`
        first, because also the result :class:`~multiprocessing.Queue`
        will be closed.
        """
        for i in range(self._nprocs):
            self._taskq.put(self._sentinel)
        self._taskq.close()
        self._resultq.close()
        self._pool_closed = True
    
    def terminate(self):
        """
        Terminate the process pool.
        
        Stops the worker processes immediately without completing
        outstanding work.  When the pool object is garbage collected
        :meth:`terminate` will be called.
        """
        for proc in self._procs:
            proc.terminate()
    
    def join(self):
        """
        Wait for the worker processes to exit.
        
        One should call :meth:`close` or :meth:`terminate` before using
        :meth:`join`.  If :meth:`join` is called without calling
        :meth:`close` or :meth:`terminate` before, :meth:`close` will be
        called internally before :meth:`join`.
        """
        if not self._pool_closed:
            self.close()
        self._taskq.join_thread()
        self._resultq.join_thread()
        for proc in self._procs:
            proc.join()
