/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_UTIL_H
#define NCCL_OFI_UTIL_H

#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>

#include "nccl_ofi_log.h"

extern char **environ;

class env_manager {
public:
	static env_manager& getInstance() {
		static env_manager instance;
		return instance;
	}

	// Prevent copying and assignment
	env_manager(const env_manager&) = delete;
	env_manager& operator=(const env_manager&) = delete;

	void insert_envvar(std::string name, std::string val, bool overwrite)
	{
		std::lock_guard<std::mutex> l(lock);

		env.emplace(name, storage(val, overwrite));
	}

	void insert_envvars(std::map<std::string, std::string> old_env)
	{
		std::lock_guard<std::mutex> l(lock);

		for (auto iter = old_env.begin() ; iter != old_env.end() ; ++iter) {
			env.emplace(iter->first, storage(iter->second, false));
		}
	}

	void update_environment()
	{
		std::lock_guard<std::mutex> l(lock);
		std::unordered_set<std::string_view> found_keys;

		// setenv() is not threadsafe relative to getenv() and in the
		// multiple accelerator per process case, PyTorch is frequently
		// calling getenv() in threads at the same time that the plugin
		// is getting initialized.  So we steal an idea from RCU in the
		// Linux kernel and copy the entire environ pointer and just
		// avoid the whole situation.  This is not threadsafe with
		// itself (or with other calls to setenv), but this should be
		// good enough to cover all the real problems we had with
		// setenv.  This also relies on getenv() being implemented in a
		// way that doesn't refer to environ every step in the loop,
		// which is true of glibc's implementation.

		size_t new_size = 0;
		while (environ[new_size] != NULL) {
			new_size++;
		}
		new_size += (env.size() + 1);

		char **new_environ = new char*[new_size + 1];

		// Copy the old environ into the new environ, except for entries
		// we know we want to replace because they are in `env`.
		size_t i_old = 0, i_new = 0;
		while (environ[i_old] != NULL) {
			std::string_view entry(environ[i_old]);

			auto delim = entry.find("=");
			auto name = entry.substr(0, delim);

			auto env_iter = env.find(std::string(name));
			if (env_iter == env.end() || !env_iter->second.overwrite) {
				new_environ[i_new] = environ[i_old];
				i_new++;
				found_keys.insert(name);
			}

			i_old++;
		}

		// Now copy in the new entries from `env`
		for (auto iter = env.begin() ; iter != env.end() ; ++iter) {
			size_t len;

			auto& name = iter->first;
			auto& value = iter->second.value;

			// skip if we found it in the environ and don't want to overwrite
			if (found_keys.find(name) != found_keys.end() &&
			    !iter->second.overwrite) {
				NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
					      "Skipping adding %s to environment, already set",
					      name.c_str());
				continue;
			}

			len = name.length() + value.length() + 2;
			// this is going in the C enviorn, need to use malloc here
			char *tmp = (char *)malloc(len);
			if (tmp == NULL) {
				throw std::bad_alloc();
			}

			memcpy(tmp, name.c_str(), name.length());
			len = iter->first.length();

			memcpy(tmp + len, "=", 1);
			len += 1;

			memcpy(tmp + len, value.c_str(), value.length());
			len += value.length();
			tmp[len] = '\0';

			NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
				      "Adding %s to environment", tmp);

			new_environ[i_new] = tmp;
			i_new++;
		}

		// and the NULL terminator entry
		new_environ[i_new] = NULL;

		// note that this is like RCU, but without the cleanup phase.
		// We just leak the old environ pointer and any entries that we
		// didn't copy.  It's small, and better than crashing.
		__sync_synchronize();
		environ = new_environ;
	}


private:
	env_manager() { }

	class storage {
	public:
		storage(const std::string& value_arg, bool overwrite_arg)
			: value(value_arg), overwrite(overwrite_arg)
		{ }

		std::string value;
		bool overwrite;
	};

	std::map<std::string, storage> env;
	std::mutex lock;
};


#endif // NCCL_OFI_UTIL_H
