/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <cstdio>

#include "unit_test.h"
#include "nccl_ofi_assert.h"
#include "nccl_ofi_dlist.h"

struct entry {
	uint32_t id;
	nccl_ofi_dlist_node node;
};

static entry *to_entry(nccl_ofi_dlist_node *n)
{
	return nccl_ofi_dlist_entry(n, &entry::node);
}

static entry *pop_entry(nccl_ofi_dlist *list)
{
	nccl_ofi_dlist_node *n = list->pop_front();
	assert_always(n != nullptr);
	return to_entry(n);
}

static void test_empty_list()
{
	nccl_ofi_dlist list;
	assert_always(list.empty());
	assert_always(list.front() == nullptr);
	assert_always(list.pop_front() == nullptr);
}

static void test_on_list()
{
	entry e = {1, {}};
	assert_always(!e.node.on_list());

	nccl_ofi_dlist list;
	list.push_back(&e.node);
	assert_always(e.node.on_list());

	e.node.remove();
	assert_always(!e.node.on_list());
}

static void test_push_back_front()
{
	nccl_ofi_dlist list;

	entry a = {1, {}}, b = {2, {}}, c = {3, {}};
	list.push_back(&a.node);
	assert_always(to_entry(list.front())->id == 1);

	list.push_back(&b.node);
	list.push_back(&c.node);
	assert_always(to_entry(list.front())->id == 1);
}

static void test_pop_front_order()
{
	nccl_ofi_dlist list;

	entry a = {1, {}}, b = {2, {}}, c = {3, {}};
	list.push_back(&a.node);
	list.push_back(&b.node);
	list.push_back(&c.node);

	assert_always(pop_entry(&list)->id == 1);
	assert_always(pop_entry(&list)->id == 2);
	assert_always(pop_entry(&list)->id == 3);
	assert_always(list.empty());
}

static void test_remove_head()
{
	nccl_ofi_dlist list;

	entry a = {1, {}}, b = {2, {}}, c = {3, {}};
	list.push_back(&a.node);
	list.push_back(&b.node);
	list.push_back(&c.node);

	a.node.remove();
	assert_always(pop_entry(&list)->id == 2);
	assert_always(pop_entry(&list)->id == 3);
	assert_always(list.empty());
}

static void test_remove_middle()
{
	nccl_ofi_dlist list;

	entry a = {1, {}}, b = {2, {}}, c = {3, {}};
	list.push_back(&a.node);
	list.push_back(&b.node);
	list.push_back(&c.node);

	b.node.remove();
	assert_always(pop_entry(&list)->id == 1);
	assert_always(pop_entry(&list)->id == 3);
	assert_always(list.empty());
}

static void test_remove_tail()
{
	nccl_ofi_dlist list;

	entry a = {1, {}}, b = {2, {}}, c = {3, {}};
	list.push_back(&a.node);
	list.push_back(&b.node);
	list.push_back(&c.node);

	c.node.remove();
	assert_always(pop_entry(&list)->id == 1);
	assert_always(pop_entry(&list)->id == 2);
	assert_always(list.empty());
}

static void test_remove_only()
{
	nccl_ofi_dlist list;

	entry a = {1, {}};
	list.push_back(&a.node);
	a.node.remove();
	assert_always(list.empty());
}

static void test_reinsert_after_remove()
{
	nccl_ofi_dlist list;

	entry a = {1, {}};
	list.push_back(&a.node);
	a.node.remove();
	assert_always(list.empty());

	a.id = 2;
	list.push_back(&a.node);
	assert_always(to_entry(list.front())->id == 2);
}

static void test_pop_front_if_match()
{
	nccl_ofi_dlist list;

	entry a = {42, {}};
	list.push_back(&a.node);

	auto *r = list.pop_front_if([](nccl_ofi_dlist_node *n) {
		return to_entry(n)->id == 42;
	});
	assert_always(r == &a.node);
	assert_always(list.empty());
}

static void test_pop_front_if_no_match()
{
	nccl_ofi_dlist list;

	entry a = {42, {}};
	list.push_back(&a.node);

	auto *r = list.pop_front_if([](nccl_ofi_dlist_node *n) {
		return to_entry(n)->id == 99;
	});
	assert_always(r == nullptr);
	assert_always(!list.empty());
}

static void test_pop_front_if_empty()
{
	nccl_ofi_dlist list;

	auto *r = list.pop_front_if([](nccl_ofi_dlist_node *) { return true; });
	assert_always(r == nullptr);
}

static void test_for_each_safe()
{
	nccl_ofi_dlist list;

	entry a = {1, {}}, b = {2, {}}, c = {3, {}};
	list.push_back(&a.node);
	list.push_back(&b.node);
	list.push_back(&c.node);

	/* Remove even-id entries during iteration */
	nccl_ofi_dlist_node *pos;
	nccl_ofi_dlist_for_each_safe(&list, pos) {
		if (to_entry(pos)->id == 2)
			pos->remove();
	}

	assert_always(pop_entry(&list)->id == 1);
	assert_always(pop_entry(&list)->id == 3);
	assert_always(list.empty());
}

static void test_for_each_safe_remove_all()
{
	nccl_ofi_dlist list;

	entry a = {1, {}}, b = {2, {}}, c = {3, {}};
	list.push_back(&a.node);
	list.push_back(&b.node);
	list.push_back(&c.node);

	nccl_ofi_dlist_node *pos;
	nccl_ofi_dlist_for_each_safe(&list, pos) {
		pos->remove();
	}

	assert_always(list.empty());
}

static void test_interleaved()
{
	nccl_ofi_dlist list;

	entry a = {1, {}}, b = {2, {}}, c = {3, {}}, d = {4, {}};
	list.push_back(&a.node);
	list.push_back(&b.node);
	list.pop_front();		/* removes a, list: [b] */
	list.push_back(&c.node);	/* list: [b, c] */
	b.node.remove();		/* list: [c] */
	list.push_back(&d.node);	/* list: [c, d] */

	assert_always(pop_entry(&list)->id == 3);
	assert_always(pop_entry(&list)->id == 4);
	assert_always(list.empty());
}

int main(int argc, char *argv[])
{
	unit_test_init();

	test_empty_list();
	test_on_list();
	test_push_back_front();
	test_pop_front_order();
	test_remove_head();
	test_remove_middle();
	test_remove_tail();
	test_remove_only();
	test_reinsert_after_remove();
	test_pop_front_if_match();
	test_pop_front_if_no_match();
	test_pop_front_if_empty();
	test_for_each_safe();
	test_for_each_safe_remove_all();
	test_interleaved();

	printf("Test completed successfully!\n");

	return 0;
}
