/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_DLIST_H
#define NCCL_OFI_DLIST_H

#include <cstddef>
#include "nccl_ofi_config_bottom.h"

/**
 * Intrusive doubly-linked list.
 *
 * Embed a nccl_ofi_dlist_node in your struct, and use nccl_ofi_dlist
 * as the list head.  All operations are O(1).
 *
 * Usage:
 *   struct my_item {
 *       int data;
 *       nccl_ofi_dlist_node link;
 *   };
 *
 *   nccl_ofi_dlist list;
 *
 *   my_item item;
 *   list.push_back(&item.link);
 *
 *   // Iterate (safe to remove current node):
 *   nccl_ofi_dlist_node *pos;
 *   nccl_ofi_dlist_for_each_safe(&list, pos) {
 *       my_item *p = nccl_ofi_dlist_entry(pos, &my_item::link);
 *   }
 */

struct nccl_ofi_dlist_node {
	nccl_ofi_dlist_node *prev = nullptr;
	nccl_ofi_dlist_node *next = nullptr;

	/* True if this node is currently linked into a list. */
	bool on_list() const { return next != nullptr; }

	void remove()
	{
		prev->next = next;
		next->prev = prev;
		prev = nullptr;
		next = nullptr;
	}
};

struct nccl_ofi_dlist {
	nccl_ofi_dlist_node head = {&head, &head};

	bool empty() const { return head.next == &head; }

	void push_back(nccl_ofi_dlist_node *node)
	{
		node->prev = head.prev;
		node->next = &head;
		head.prev->next = node;
		head.prev = node;
	}

	/* Return the first node, or nullptr if empty. */
	nccl_ofi_dlist_node *front() { return empty() ? nullptr : head.next; }

	/* Remove and return the first node, or nullptr if empty. */
	nccl_ofi_dlist_node *pop_front()
	{
		if (empty())
			return nullptr;
		nccl_ofi_dlist_node *node = head.next;
		node->remove();
		return node;
	}

	/* Remove and return the first node if pred returns true,
	   or nullptr if empty or pred returns false. */
	nccl_ofi_dlist_node *pop_front_if(bool (*pred)(nccl_ofi_dlist_node *))
	{
		if (empty() || !pred(head.next))
			return nullptr;
		return pop_front();
	}
};

/* Recover the containing struct from a node pointer.
   Uses cpp_container_of for safety with non-POD types. */
template <class Parent>
Parent *nccl_ofi_dlist_entry(nccl_ofi_dlist_node *ptr, nccl_ofi_dlist_node Parent::*member)
{
	return cpp_container_of(ptr, member);
}

/**
 * Safe iteration: allows removal of the current node during the loop.
 * @list:  nccl_ofi_dlist * list
 * @pos:   nccl_ofi_dlist_node * loop cursor
 */
#define nccl_ofi_dlist_for_each_safe(list, pos) \
	for (nccl_ofi_dlist_node *pos##_next_ = \
		((pos) = (list)->head.next, (pos)->next); \
	     (pos) != &(list)->head; \
	     (pos) = pos##_next_, pos##_next_ = (pos)->next)

#endif
