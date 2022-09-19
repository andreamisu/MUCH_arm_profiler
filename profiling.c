#define _GNU_SOURCE             /* See feature_test_macros(7) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>


// #include "performance_counters.h"


#define __NR_perf_event_open 241
/// Indicates which thread/process performance counters to follow.
#define this_thread 0

/// Enables monitoring of the task performance events on any cores
#define any_core -1


/** @brief Struct holding raw measurement and ID of a performance counter.
 *
 */
struct event {
	long unsigned value; /* The value of the event */
	long unsigned id; /* if PERF_FORMAT_ID */
};

/** @brief Struct returned by the kernel upon reading the file descriptor of the performance counters.
 *  @details The struct holds values for l1-D refills and misses and l2 refills and misses.
 */
struct read_format {
	long unsigned nr; /* The number of events */
	long unsigned time_enabled; /* if PERF_FORMAT_TOTAL_TIME_ENABLED */
	long unsigned time_running; /* if PERF_FORMAT_TOTAL_TIME_RUNNING */
	struct event pmu_event_1;
	struct event pmu_event_2;
	struct event pmu_event_3;
	struct event pmu_event_4;
	struct event pmu_event_5;
	struct event pmu_event_6;
};


struct perf_counters {
    long unsigned pmu_event_1;
    long unsigned pmu_event_2;
    long unsigned pmu_event_3;
    long unsigned pmu_event_4;
	long unsigned pmu_event_5;
	long unsigned pmu_event_6;
};


//pca bench

/// Mean vector
static float *mean;
/// Standard deviation vector
static float *stddev;
/// Correlation matrix
static float **symmat;
/// Copy of correlation matrix
static float **symmat2;
/// Input data matrix
static float **data;
/// Vector of eigenvalues
static float *evals;
/// Intermediate dummy vector
static float *interm;
static char option;
/// Number of input rows
static int n;
/// Number of input columns
//static int m;



// log int 

// array of a
static int * a;

// array of a
static int * b;

// array of a
static int * m;

// thread allocated
static int thread = 1;

// iterations of the bench
static int iterations = 100000;

// iterations of the bench
static int priority = -1;

// iterations of the bench
static int isolcpu =-1;




/// Performance counters at the start of the period.
static struct perf_counters job_perf_counters_start;

/// Performance counters at the end of the period.
static struct perf_counters job_perf_counters_end;
/// File descriptor for L1-D references (also, group-fd head)
static int pmu_event_1_fd;

/// File descriptor for L1-D missess
static int pmu_event_2_fd;

/// File descriptor for L2 references
static int pmu_event_3_fd;

/// File descriptor for L2 misses
static int pmu_event_4_fd;

/// File descriptor for instruction retired
static int pmu_event_5_fd;

/// File descriptor for clock cycles count
static int pmu_event_6_fd;

/// array list of PMU to be allocated
static int*  pmu_array;

/**
 * @brief Open a file descriptor for the performance counter specified.
 * @param[in] pmc_type Specify the event type.
 * @param[in] pmc_config The platform specific ID of the performance counter.
 * @param[in] group_fd The file descriptor group to which the performance counter belongs.
 * @return The file directory opened, -1 on failures.
 */
static int open_pmc_fd(unsigned int pmc_type, unsigned int pmc_config, int group_fd)
{
	static struct perf_event_attr attr;
	attr.type = pmc_type;
	attr.config = pmc_config;
	attr.size = sizeof(struct perf_event_attr);
	attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID |
			   PERF_FORMAT_TOTAL_TIME_ENABLED |
			   PERF_FORMAT_TOTAL_TIME_RUNNING;

	int fd = syscall(__NR_perf_event_open, &attr, this_thread, any_core,
			 group_fd, 0);

	if (fd == -1) {
		perror("Could not open fd for performance counter\n");
		exit(EXIT_FAILURE);
	}

	return fd;
}


int setup_pmcs(void)
{
	//elogf(LOG_LEVEL_TRACE, "Opening performance counters fd\n");
	pmu_event_1_fd = open_pmc_fd(PERF_TYPE_RAW, pmu_array[0], -1);
	if (pmu_event_1_fd == -1)
		return -1;
	pmu_event_2_fd = open_pmc_fd(PERF_TYPE_RAW, pmu_array[1], pmu_event_1_fd);
	if (pmu_event_2_fd == -1)
		return -1;
	pmu_event_3_fd = open_pmc_fd(PERF_TYPE_RAW, pmu_array[2], pmu_event_1_fd);
	if (pmu_event_3_fd == -1)
		return -1;
	pmu_event_4_fd = open_pmc_fd(PERF_TYPE_RAW, pmu_array[3], pmu_event_1_fd);
	if (pmu_event_4_fd == -1)
		return -1;
	pmu_event_5_fd = open_pmc_fd(PERF_TYPE_RAW, pmu_array[4], pmu_event_1_fd);
	if (pmu_event_5_fd == -1)
		return -1;
	pmu_event_6_fd = open_pmc_fd(PERF_TYPE_RAW, pmu_array[5], pmu_event_1_fd);
        if (pmu_event_6_fd == -1)
                return -1;
	return pmu_event_1_fd;
}


/**
 * @brief Close the file descriptor related to the performance counters.
 * @param[in] fd The file descriptor to close.
 * @return Returns file descriptor status upon closing, return -1 on failures.
 */
static inline int close_pmc_fd(int fd)
{
	int ret = close(fd);
	if (ret == -1) {
		perror("Could not close fd for performance counter\n");
	}
	return ret;
}

/** @brief Close access to performance counters.
 * @return 0 on sucess, -1 on error.
 */
int teardown_pmcs(void)
{
	//elogf(LOG_LEVEL_TRACE, "Closing performance counters fd\n");
	int ret = 0;
	ret = close_pmc_fd(pmu_event_1_fd);
	if (ret == -1)
		return ret;
	ret = close_pmc_fd(pmu_event_2_fd);
	if (ret == -1)
		return ret;
	ret = close_pmc_fd(pmu_event_3_fd);
	if (ret == -1)
		return ret;
	ret = close_pmc_fd(pmu_event_4_fd);
	if (ret == -1)
		return ret;
	ret = close_pmc_fd(pmu_event_5_fd);
	if (ret == -1)
		return ret;
	ret = close_pmc_fd(pmu_event_6_fd);
        if (ret == -1)
                return ret;
	return 0;
}


struct perf_counters pmcs_get_value(void)
{
	struct read_format measurement;
	size_t size = read(pmu_event_1_fd, &measurement,
			   sizeof(struct read_format));
	if (size != sizeof(struct read_format)) {
		perror("Error: Size read from performance counters differ from size expected.");
	}
	struct perf_counters res;
	res.pmu_event_1 = measurement.pmu_event_1.value;
	res.pmu_event_2 = measurement.pmu_event_2.value;
	res.pmu_event_3 = measurement.pmu_event_3.value;
	res.pmu_event_4 = measurement.pmu_event_4.value;
	res.pmu_event_5 = measurement.pmu_event_5.value;
	res.pmu_event_6 = measurement.pmu_event_6.value;
	return res;
}

// initial benchmark
long loop_count,l_c, th_count;
struct timeval t;
double f_avg, i_avg;

// Thread Structure for FLOPS
struct fth
{
	int lc, th_counter;
	float fa, fb, fc, fd;
	pthread_t threads;
};


void *FAdd(struct fth *data)
{
	for(data->lc = 1; data->lc <= 100000000; data->lc++)
	{
		data->fb + data->fc;
		data->fa - data->fb;
		data->fa + data->fd;
		data->fa + data->fb;
		data->fb + data->fc;
		data->fa - data->fb;
		data->fa + data->fd;
		data->fa + data->fb;
		data->fb + data->fc;
		data->fa - data->fb;
		data->fb + data->fc;
		data->fa - data->fb;
		data->fa + data->fd;
		data->fa + data->fb;
		data->fb + data->fc;
		data->fa - data->fb;
		data->fa + data->fd;
		data->fa + data->fb;
		data->fb + data->fc;
		data->fa - data->fb;
		data->fb + data->fc;
		data->fa - data->fb;
		data->fa + data->fd;
		data->fa + data->fb;
		data->fb + data->fc;
		data->fa - data->fb;
		data->fa + data->fd;
		data->fa + data->fb;
		data->fb + data->fc;
		data->fa - data->fb;
	}
}


int discreteLogarithm(int a, int b, int m)
{
	int n = (int) (sqrt (m) + 1);

	// Calculate a ^ n
	int an = 1;
	for (int i = 0; i < n; ++i)
		an = (an * a) % m;

	int *value = calloc(m,sizeof(int));

	// Store all values of a^(n*i) of LHS
	for (int i = 1, cur = an; i <= n; ++i)
	{
		if (value[ cur ] == 0)
			value[ cur ] = i;
		cur = (cur * an) % m;
	}

	for (int i = 0, cur = b; i <= n; ++i)
	{
		// Calculate (a ^ j) * b and check
		// for collision
		if (value[cur] > 0)
		{
			int ans = value[cur] * n - i;
			if (ans < m){
				free(value);
				return ans;
			}
				
		}
		cur = (cur * a) % m;
	}
	free(value);
	return -1;
}

void benchmarkSetup(){
	srand(time(NULL));
	int upper = 50;
	int lower = 10;
	a = (int*)calloc(iterations, sizeof(int));
	b = (int*)calloc(iterations, sizeof(int));
	m = (int*)calloc(iterations, sizeof(int));
	for(int i; i<iterations;i++){
		a[i] = (rand() % (upper - lower + 1)) + lower;
		b[i] = (rand() % (50 - 10 + 1)) + 10;
		m[i] = (rand() % (100 - 51 + 1)) + 51;
	}
}

// void main(){
// 	srand(time(NULL));   // Initialization, should only be called once.
// 	int a = 2, b = 3, m = 5;
// 	int ret = discreteLogarithm(a, b, m);
// 	printf("discreteLog: %d\n" , ret);
	
// 	int upper = 50;
// 	int lower = 10;
// 	a = (rand() % (upper - lower + 1)) + lower;
// 	b = (rand() % (upper - lower + 1)) + lower;
// 	m = (rand() % (upper - lower + 1)) + lower;
// 	ret = discreteLogarithm(a, b, m);
// 	printf("a: %d\nb: %d\nm: %d\n" , a,b,m);
// 	printf("discreteLog: %d\n" , ret);
// }

void FLOPSBenchmark(struct fth * ft)
{
    FAdd(ft);
}

void*  logarithmBenchmark(){
	for(int i=0;i<iterations;i++){
		discreteLogarithm(a[i], b[i], m[i]);
	}
}

int main(int argc, char *argv[]) {
    size_t optind;
	pthread_t* threads;

	if(argc<2){
		fprintf(stderr, "Usage: %s [PMUs] [-t] [threads] [-i] [iterations] [-c] [isolcpu] [p] [priority]\n", argv[0]);
		exit(EXIT_FAILURE);
	}
	
	for (optind = 1; optind < argc; optind++) {
		if(argv[optind][0] != '-')
			continue;
		switch (argv[optind][1]) {
			case 't': thread = atoi(argv[optind+1]); break;
			case 'i': iterations = atoi(argv[optind+1]); break;
			case 'c': isolcpu = atoi(argv[optind+1]); break;
			case 'p': priority = atoi(argv[optind+1]); break;
			default:
				fprintf(stderr, "Usage: %s [PMUs] [-t] [threads] [-i] [iterations] [-c] [isolcpu] [-p] [priority]\n", argv[0]);
				exit(EXIT_FAILURE);
		}   
	}
	char *pmus = strtok(argv[1], "/");

	if(isolcpu!=-1){
		cpu_set_t my_set;
		CPU_ZERO(&my_set); 
		CPU_SET(isolcpu, &my_set); 
		sched_setaffinity(0, sizeof(cpu_set_t), &my_set);
	}

	if(priority!=-1){
		int which = PRIO_PROCESS;
		id_t pid;
		int ret;

		pid = getpid();
		ret = getpriority(which, pid);
		printf("priority was: %d\n", ret);

		ret = setpriority(which, pid, priority);
		printf("priority now is: %d\n", ret);
	}
	
    pmu_array = calloc(6,sizeof(int));
    int index = 0;

    while(pmus != NULL && index<6)
	{
        //printf("found: %s\n", pmus);
        pmu_array[index++] = (int)strtol(pmus, NULL, 0);
        //printf("pmu_array: %d\n", pmu_array[index-1]);
		pmus = strtok(NULL,  "/");
	}
    //SETUP PMUs
    int group_fd = setup_pmcs();

    //SETUP bench
	if(thread>1){
		threads = (pthread_t*)malloc(thread * sizeof(pthread_t));
	}
    benchmarkSetup(iterations);

    //INIT PMU VALUES 
    job_perf_counters_start = pmcs_get_value();

    //START BENCHMARK
    //SETUP bench
	if(thread>1){
		for(int i=0;i<thread;i++){
			if(pthread_create(&threads[i], NULL, logarithmBenchmark, NULL)){
				printf("benchmark has failed. aborting.");
				exit(EXIT_FAILURE);
			}
		}
		for(int i=0;i<thread;i++){
			pthread_join(threads[i], NULL);
		}
	}
	else
		logarithmBenchmark();


    //STOP PMU VALUES
    job_perf_counters_end = pmcs_get_value();


    // printf("job_perf_counters_start.pmu_event_1: %lu\n", job_perf_counters_start.pmu_event_1);
    // printf("job_perf_counters_start.pmu_event_2: %lu\n", job_perf_counters_start.pmu_event_2);
    // printf("job_perf_counters_start.pmu_event_3: %lu\n", job_perf_counters_start.pmu_event_3);
    // printf("job_perf_counters_start.pmu_event_4: %lu\n", job_perf_counters_start.pmu_event_4);
    // printf("job_perf_counters_start.pmu_event_5: %lu\n", job_perf_counters_start.pmu_event_5);
    // printf("job_perf_counters_start.pmu_event_6: %lu\n", job_perf_counters_start.pmu_event_6);
    // printf("job_perf_counters_end.pmu_event_1: %lu\n", job_perf_counters_end.pmu_event_1);
    // printf("job_perf_counters_end.pmu_event_2: %lu\n", job_perf_counters_end.pmu_event_2);
    // printf("job_perf_counters_end.pmu_event_3: %lu\n", job_perf_counters_end.pmu_event_3);
    // printf("job_perf_counters_end.pmu_event_4: %lu\n", job_perf_counters_end.pmu_event_4);
    // printf("job_perf_counters_end.pmu_event_5: %lu\n", job_perf_counters_end.pmu_event_5);
    // printf("job_perf_counters_end.pmu_event_6: %lu\n", job_perf_counters_end.pmu_event_6);


    //TEARDOWN
    teardown_pmcs();
	if(thread>1)
		free(threads);
	free(pmu_array);
	free(a);
	free(b);
	free(m);

    //TODO: using stdout, shall we use another FD? 
    printf("%lu/%lu/%lu/%lu/%lu/%lu",
        job_perf_counters_end.pmu_event_1 - job_perf_counters_start.pmu_event_1,
        job_perf_counters_end.pmu_event_2 - job_perf_counters_start.pmu_event_2, 
        job_perf_counters_end.pmu_event_3 - job_perf_counters_start.pmu_event_3, 
        job_perf_counters_end.pmu_event_4 - job_perf_counters_start.pmu_event_4,
        job_perf_counters_end.pmu_event_5 - job_perf_counters_start.pmu_event_5,
        job_perf_counters_end.pmu_event_6 - job_perf_counters_start.pmu_event_6 );
    exit(0);
}






