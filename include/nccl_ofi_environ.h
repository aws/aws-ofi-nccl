/*
 * Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_ENVIRON_H
#define NCCL_OFI_ENVIRON_H

#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

#include "nccl_ofi_log.h"

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
		std::lock_guard l(lock);

		if (environ_frozen) {
			throw std::runtime_error("insert_envvar(" + name + ") called after environ frozen");
		}

		env.emplace(name, storage(val, overwrite));
	}

	void insert_envvars(std::map<std::string, std::string> added_env)
	{
		std::lock_guard l(lock);

		if (environ_frozen) {
			throw std::runtime_error("insert_envvars() called after environ frozen");
		}

		for (auto iter = added_env.begin() ; iter != added_env.end() ; ++iter) {
			env.emplace(iter->first, storage(iter->second, false));
		}
	}

	void update_environment(char ***environ_p)
	{
		std::lock_guard l(lock);
		std::unordered_set<std::string_view> found_keys;
		bool modified_environ = false;

		environ_frozen = true;

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
		while ((*environ_p)[new_size] != NULL) {
			new_size++;
		}
		new_size += (env.size() + 1);

		// this is going in the C environ, need to use malloc here
		char **new_environ = (char **)malloc(sizeof(char *) * (new_size + 1));
		if (OFI_UNLIKELY(new_environ == NULL)) {
			throw std::bad_alloc();
		}

		// Copy the old environ into the new environ, replacing any
		// entries that are in the new environment map.
		size_t idx = 0;
		while ((*environ_p)[idx] != NULL) {
			std::string_view entry((*environ_p)[idx]);

			auto delim = entry.find("=");
			auto name = entry.substr(0, delim);

			auto env_iter = env.find(std::string(name));
			if (env_iter == env.end()) {
				new_environ[idx] = (*environ_p)[idx];
			} else if (!env_iter->second.overwrite) {
				// found an entry, but not supposed to
				// overwrite.  Use the old value.
				NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
					      "Skipping adding %s to environment, already set",
					      name.data());
				new_environ[idx] = (*environ_p)[idx];
				modified_environ = true;
			} else if (env_iter->second.overwrite) {
				new_environ[idx] = create_new_entry(env_iter);
			}

			found_keys.insert(name);
			idx++;
		}

		// Now copy in the new entries from `env`
		for (auto iter = env.begin() ; iter != env.end() ; ++iter) {
			auto& name = iter->first;

			// skip if we found it in the environ, as already handled above
			if (found_keys.find(name) != found_keys.end()) {
				continue;
			}

			new_environ[idx] = create_new_entry(iter);
			modified_environ = true;
			idx++;
		}

		// and the NULL terminator entry
		new_environ[idx] = NULL;

		if (modified_environ) {
			// note that this is like RCU, but without the cleanup
			// phase.  We just leak the old environ pointer and any
			// entries that we didn't copy.  It's small, and better
			// than crashing.
			__sync_synchronize();
			(*environ_p) = new_environ;
		} else {
			// we didn't change the environ, so wind back our
			// changes.  Since we didn't modify the environ, we know
			// that the only memory we allocated is the "new"
			// environ array, which we don't need.  So free that and
			// we're done cleaning up.
			free(new_environ);
		}
	}


protected:
	class storage {
	public:
		storage(const std::string& value_arg, bool overwrite_arg)
			: value(value_arg), overwrite(overwrite_arg)
		{ }

		std::string value;
		bool overwrite;
	};

	typedef std::map<std::string, storage> env_storage;

	env_manager() : environ_frozen(false) { }

	char *create_new_entry(env_storage::iterator iter)
	{
		size_t len;
		auto& name = iter->first;
		auto& value = iter->second.value;

		len = name.length() + value.length() + 2;
		// this is going in the C environ, need to use malloc here
		char *tmp = (char *)malloc(sizeof(char) * len);
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

		return tmp;
	}

	bool environ_frozen;
	env_storage env;
	std::mutex lock;
};


#endif // NCCL_OFI_ENVIRON_H
