/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi_assert.h"
#include "nccl_ofi_environ.h"
#include "test-logger.h"


// test_env_manager exists only to remove the singleton restriction for unit
// tests.
class test_env_manager : public env_manager {
public:
	test_env_manager() { }
};


static void no_change_check()
{
	char **env = (char **)malloc(2 * sizeof(char *));
	char *val0 = strdup("hello=bye");
	env[0] = val0;
	env[1] = NULL;
	char **orig_envp = env;
	test_env_manager envmgr;

	envmgr.update_environment(&env);
	assert_always(env == orig_envp);
	assert_always(env[0] == val0);
}


static void addition_check()
{
	char **env = (char **)malloc(2 * sizeof(char *));
	char *val0 = strdup("hello=bye");
	env[0] = val0;
	env[1] = NULL;
	char ***orig_envp = &env;
	test_env_manager envmgr;

	envmgr.insert_envvar("womp", "womp", false);

	envmgr.update_environment(&env);
	assert_always(&env == orig_envp);
	assert_always(env[0] == val0);
	assert_always(0 == strcmp(env[1], "womp=womp"));
}


static void late_check()
{
	char **env = (char **)malloc(1 * sizeof(char *));
	env[0] = NULL;
	bool raised = false;
	test_env_manager envmgr;

	envmgr.update_environment(&env);

	try {
		envmgr.insert_envvar("womp", "wompp", false);
	} catch (...) {
		raised = true;
	}
	assert_always(raised);
}


static void replace_check()
{
	char **env = (char **)malloc(5 * sizeof(char *));
	char *val0 = strdup("one=1");
	char *val1 = strdup("two=4");
	char *val2 = strdup("three=3");
	char *val3 = strdup("four=4");
	env[0] = val0;
	env[1] = val1;
	env[2] = val2;
	env[3] = val3;
	env[4] = NULL;
	char ***orig_envp = &env;
	test_env_manager envmgr;

	envmgr.insert_envvar("two", "2", true);
	envmgr.insert_envvar("three", "5", false);
	envmgr.insert_envvar("five", "5", false);

	envmgr.update_environment(&env);
	assert_always(&env == orig_envp);
	assert_always(env[0] == val0);
	assert_always(0 == strcmp(env[1], "two=2"));
	assert_always(env[2] == val2);
	assert_always(env[3] == val3);
	assert_always(0 == strcmp(env[4], "five=5"));
	assert_always(env[5] == NULL);
}



int main(int argc, char *argv[])
{
	ofi_log_function = logger;

	no_change_check();
	addition_check();
	late_check();
	replace_check();

	return 0;
}
