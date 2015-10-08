import os
import time
import threading
import Queue as queue
from multiprocessing import Process, Queue, Value
from generators.randomparameters import RandomParameters

def evaluate_worker(thread_number, classname, paramQueue, resultQueue, finished, maxit):
    """
    evaluate_worker is a function that is executed in a new process. 
    It evaluates the performance of the algorithm executed by the given
    classname, that should be an implementation of ParameterTester, on
    sets of parameters provided through the paramQueue queue, as a
    tuple (id, params) where ID is some way to separate several parameter
    evolution algorithms. The results are put into the resultQueue as a
    tuple (id, params, result) where result is the value returned by
    the algorithm.
    """
    #print "Worker %d (re)started" % thread_number
    it = 0
    while not finished.value and it < maxit:
        try:
            id, param_set, parameters = paramQueue.get_nowait()
            # print "Evaluating params %s" % repr(parameters)
            resultQueue.put((id, thread_number, param_set, parameters, None))
            time.sleep(0.001) # Give feeder thread time to send data
            test = classname()
            test.set_parameters(parameters)
            test.run_algorithm()
            result = test.get_result()
            resultQueue.put((id, thread_number, param_set, parameters, result))
            time.sleep(0.001) # Give feeder thread time to send data
            it += 1
        except queue.Empty:
            time.sleep(0.1)

class PerformTest(threading.Thread):
    """
    PerformTest class is the class that actually evaluates the algorithm by
    instantiating a generator, generating new sets of parameters and passing
    those to the worker threads. PerformTest itself is implemented as a
    Thread subclass to keep the GUI responsive while running tests.

    Each evaluation is performed by a worker process to allow for true multi-
    processing. Also, when the algorithm evaluated crashes for some reason,
    the process can be terminated and restarted without the entire 
    GUI going down.
    """
    def __init__(self):
        """
        The constructor initializes all required datamembers
        """
        super(PerformTest, self).__init__()
        self._generator = None
        self._tester_class = None
        self._threads = None
        self._values = None
        self._callback = None
        self._running = True
        self._evaluating = False

        self._testers = []
        self._tester_params = []
        self._tester_activity = []
        self._tester_lock = threading.Lock()
        self._processing_timeout = 30
        self._max_iterations = 1000

        self._global_best = None

        self.start()

    def set_options(self, generator, tester_class, threads, num_values, callback=None, processing_timeout=30):
        """
        set_options can be used to set the options of the evaluating after
        the thread has been started
        """
        self._generator = generator
        self._generator.set_num_values(num_values)
        self._tester_class = tester_class
        self._threads = threads
        self._values = num_values
        self._callback = callback
        self._processing_timeout = processing_timeout
        path = self._tester_class.param_path
        full_cmd = self._tester_class.start_command
        if full_cmd.find(' ') != -1:
            cmd = full_cmd[0:full_cmd.find(' ')]
        else:
            cmd = full_cmd

        # Validate the given values, and do not start if anything is incorrect
        if full_cmd.find("%") != -1:
            try:
                print full_cmd
                exec_cmd = full_cmd % {'filename': 'config_file'}
            except KeyError:
                return "Warning: start command can only contain filename variable"
            except ValueError:
                return "Warning: command `%s' uses malformed formatting" % full_cmd

        if not os.path.exists(cmd):
            return "Warning: executable `%s' does not exist." % cmd
        if not os.access(cmd, os.W_OK):
            return "Warning: file `%s' is not executable." % cmd
        if not os.path.exists(path):
            return "Warning: configuration path `%s' does not exist." % path
        if not os.access(path, os.W_OK):
            return "Warning: configuraiton path `%s' is not writable." % path

        try:
            testergen = RandomParameters()
            self._tester_class.add_parameters(testergen)
            id, params = testergen.get_parameters()
            configfile = self._tester_class.get_config(params)
            return True
        except KeyError as missing_keys:
            return "Warning: invalid parameter used in configuration: %s" % missing_keys[0]
        except ValueError:
            return "Warning: malformed format string in configuration format"

        
    def run(self):
        """
        run is the main loop used by the thread. If evaluating is requested,
        it will start the evaluation, and otherwise it will idle and wait
        for commands
        """
        while self._running:
            if self._evaluating:
                self.evaluate_algorithm()
            else:
                time.sleep(0.001)

    def start_evaluation(self):
        """
        start_evaluation sets the evaluating flag to true, indicating
        that the algorithm should be evaluated using the options
        currently set
        """
        self._evaluating = True

    def evaluate_algorithm(self):
        """
        evaluate_algorithm evaluates the algorithm executed by the given 
        classname tester_class, that should be an implementation of 
        ParameterTester. It obtains the parameters from a generator that
        should be an implementation of ParameterGenerator.
        The algorithm can be evaluated with multiple processes at the same time
        by specifiying the amount of threads. This many processes will be started
        and maintained while there are still more sets of parameters to test.
        num_values specifies the amount of parameter sets to test.
        """

        # Set up everything for the evaluation
        self.__initialize_evaluation()
        self.__initialize_processes()

        # Enqueue all parameter sets
        self.__enqueue_data()

        # Wait for all the parameters to be processed
        self.__await_results()

        # Clean up all processes
        self.__stop_processes()
        self.__update_best(True)
        self._evaluating = False
        print "Done evaluating algorithm"

    def stop_evaluation(self):
        """
        stop_evaluation sets the evaluating flag to false, indicating
        that the class should stop evaluating the algorithm as soon
        as possible.
        """
        if not self._evaluating:
            return
        self.__stop_processes()
        self._evaluating = False

    def stop_running(self):
        """
        stop_running stops the PerformTest thread. It first sets the
        evaluating status to False and then waits for the current
        evaluation to end
        """
        self.stop_evaluation()
        self._running = False
        self.join()

    def stop_evaluation(self):
        if not self._evaluating:
            return
        self._evaluating = False

    #####################
    ## PRIVATE METHODS ##
    #####################
    def __initialize_evaluation(self):
        """
        Set up the datamembers for evaluation, such as
        the queues and the shared values.
        """
        # Make sure that the lists are of proper length
        self._evaluating = True
        self.paramQueue = Queue(self._threads * 4)
        self.resultQueue = Queue(self._threads * 4)
        self.finished = Value('b', False)
        self.finished.value = False

    def __initialize_processes(self):
        """
        Start the set amount of worked processes to evaluate
        the algorithm
        """
        with self._tester_lock:
            self._testers = []
            self._tester_params = []
            self._tester_activity = []
            for i in range(self._threads):
                p = Process(target=evaluate_worker, args=(i, self._tester_class, self.paramQueue, self.resultQueue, self.finished, self._max_iterations))
                p.start()
                self._testers.append(p)
                self._tester_activity.append(time.time())
                self._tester_params.append([])

    def __enqueue_data(self):
        """
        Generate all data using the set generator and put it
        in the processing queue. Meanwhile, keep checking if
        any results are back in and update the frontend and
        the parameter generator with those results
        """
        for num in range(self._values):
            # Keep trying to enqueue the next data set until it succeeds
            while self._running and self._evaluating:
                if self._generator.algorithm_done:
                    print "[PerformTest] Generator algorithm finished, no more parameter sets"
                    if self._callback:
                        self._callback("GENERATOR_FINISHED", (0, {}, 0), 0)
                    else:
                        print "[PerformTest] Generator algorithm finished, no more parameter sets"
                    break
                
                self.__process_results()
                self.__check_process_activity()
                if self.__enqueue_parameters(num):
                    break # Enqueuing succeeded
                time.sleep(0.001)
        print "Done enqueing data"

    def __await_results(self):
        """
        Wait for all parameter sets to be removed from the parameter queue, processed
        and succesfully returned through the result queue.
        """
        processing = True
        while (not self.paramQueue.empty() or processing) and self._running and self._evaluating:
            self.__check_process_activity()
            self.__process_results()
            processing = False
            # Check if any parameter set is still being evaluated, and if so,
            # keep looping until everything has been processed
            for idx in range(self._threads):
                with self._tester_lock:
                    if self._tester_params[idx]:
                        processing = True
                        print "Thread %d is still processing: %s" % (idx, repr(self._tester_params[idx]))
            time.sleep(0.1)
        self.finished.value = True
        print "All values finished processing"

    def __stop_processes(self):
        """
        Stop all processes, either the hard or the soft way, depending
        on whether self._running and self._evaluating are true or not. If
        any of them is false, immediate termination is requested and thus
        the processes will be terminated. Otherwise, they will simply be
        joined, allowing them to complete their work.
        """
        for idx in range(self._threads):
            with self._tester_lock:
                if self._testers[idx]:
                    if not self._running or not self._evaluating:
                        if self._callback:
                            self._callback(idx, "[%d] Terminating thread" % (idx + 1), 0)
                        else:
                            print "[%d] Terminating thread" % (idx + 1)
                        self._testers[idx].terminate()
                    else:
                        print "[%d] Joining thread" % (idx + 1)
                        self._testers[idx].join()
                        print "[%d] Thread finished" % (idx + 1)
                    self._testers[idx] = None
                    self._tester_params[idx] = None
        print "All processes stopped"

    def __process_results(self):
        """
        Check if any results have been posted to the resultQueue by the worker
        processes, and if so, pass them back to the parameter generator.
        """
        try:
            # This will throw an exception if no data is available
            param_id, thread, param_set, params, result = self.resultQueue.get(timeout=0.1)
            if result != None:
                # If the result is not None, this is a notification that a process has
                # finished evaluating a parameter set.

                # Remove the item from the currently processing list
                with self._tester_lock:
                    self._tester_params[thread] = None
                    self._tester_activity[thread] = time.time()

                # Pass the result back to the parameter generator. This will, if it choses to
                # return the score and parameters to present to the frontend. If it
                # returns None, no best score will be passed to the frontend.
                corrected_result = self._generator.set_result(params, result, id=param_id)
                if corrected_result:
                    params, result = corrected_result
                    self.__update_best()
                if self._callback:
                    self._callback(thread, "[%d] Finished evalution of set %d of %d - score: %.5f" % (thread + 1, param_set + 1, self._values, result), param_set)
                else:
                    print "[%d] Finished evalution of set %d of %d - score: %.5f" % (thread + 1, param_set + 1, self._values, result)
            else:
                # If the result is None, this is a notification that a process has begun to
                # process a parameter set.
                item = (param_id, param_set, params)
                with self._tester_lock:
                    self._tester_activity[thread] = time.time()
                    self._tester_params[thread] = item
                if self._callback:
                    self._callback(thread, "[%d] Evaluating set %d of %d" % (thread + 1, param_set + 1, self._values), 0)
                else:
                    print "[%d] Evaluating set %d of %d" % (thread + 1, param_set + 1, self._values)
        except queue.Empty:
            pass

    def __enqueue_parameters(self, num):
        """
        This enqueues a paremeter set, if possible. It will not enqueue
        more than two of times the number of threads if the qsize parameter
        is available. Otherwise, problems might occur even though
        this is unlikely, because the paramater generator probably will
        not be able to keep up as it needs to get results of earlier
        parameter sets back first.
        """
        do = False
        try:
            size = self.paramQueue.qsize()
            if size < self._threads * 4:
                do = True
        except:
            do = True
        if do:
            id, params = self._generator.get_parameters()
            # See if a new parameter set has been returned
            if params:
                self.paramQueue.put((id, num, params))
                time.sleep(0.01) # Give feeder thread time to send data
                return True
        return False

    def __check_process_activity(self):
        """
        Check each process to see if it is still active. If it has been
        processing a single parameter set for too long, the process will be
        terminated and a new process will be started. The parameter set the
        process was processing will be enqueued again.
        """
        for idx in range(self._threads):
            with self._tester_lock:
                if self._tester_activity[idx] < time.time() - self._processing_timeout:
                    if self._tester_params[idx]:
                        # This process should be processing some set, but has not
                        # responded for too long. 
                        id, param_set, params = self._tester_params[idx]
                        self._tester_params[idx] = None

                        # Kill and restart process
                        self.__restart_process(idx)

                        # Re-enqueue the data the killed process was working on
                        self.paramQueue.put((id, param_set, params))
                        time.sleep(0.1) # Give feeder thread time to send data
                        if self._callback:
                            self._callback(idx, "[%d] Evaluating set %d of %d took too long, thread terminated" % (idx + 1, param_set + 1, self._values), 0)
                        else:
                            print "[%d] Evaluating set %d of %d took too long, thread terminated" % (idx + 1, param_set + 1, self._values)

                if not self._testers[idx] or not self._testers[idx].is_alive():
                    self.__restart_process(idx)
                    if self._callback:
                        self._callback(idx, "[%d] Thread has performed over %d iterations, restarting process" % (idx + 1, self._max_iterations), 0)
                    else:
                        print "[%d] Thread has performed over %d iterations, restarting process" % (idx + 1, self._max_iterations)

    def __restart_process(self, idx):
        # If we are finished or not running, no need to restart processes
        if self._testers[idx] and self._testers[idx].is_alive():
            self._testers[idx].terminate()
        self._testers[idx] = None

        if self.finished.value or not self._evaluating:
            return

        self._testers[idx] = Process(target=evaluate_worker, args=(idx, self._tester_class, self.paramQueue, self.resultQueue, self.finished, self._max_iterations))
        self._testers[idx].start()
        self._tester_activity[idx] = time.time()

    def __update_best(self, finished=False):
        """
        Update the best value in the frontend, by retrieving the best value
        from the parameter generator.
        """
        best_params, best_score = self._generator.get_best_result()

        # If there is a parameter set to present, send it to the frontend
        if best_params != None:
            if self._global_best != None and not finished:
                improvement = abs(best_score - self._global_best)
                if improvement < 0.00000001:
                    return

            self._global_best = best_score
            self.__write_best(best_score, best_params)
            if self._callback:
                if finished: 
                    print "Sending finished to callback"
                    self._callback("FINISHED", (0, best_params, best_score), 0)
                else:
                    self._callback("BEST", (0, best_params, best_score), 0)
            elif finished:
                print "\nBest result: "
                self._generator.present_result(best_params, best_score)

    def __write_best(self, score, params):
        """
        This method writes a score to the Results directory
        """
        program_location = os.path.dirname(os.path.abspath(__file__))
        os.chdir(program_location)

        filename = "Results/Score_%s_%s_%d_Evals_Score_%.6f" % (self._tester_class.__name__, self._generator.__class__.__name__, self._values, score)
        tester_obj = self._tester_class()
        print "Now writing best results to %s" % filename
        tester_obj.set_parameters(params)
        tester_obj.write_parameters(filename)
