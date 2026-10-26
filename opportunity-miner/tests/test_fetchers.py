"""Fetcher parsers and one mocked HTTP pass per source."""

from __future__ import annotations

import unittest

import httpx

from tests.support import ROOT  # noqa: F401

from config import Settings
from ingestion.feeds import parse_feed
from ingestion.hn_fetcher import HackerNewsFetcher, dedupe_ids, fetch_hn_stories, normalize_story
from ingestion.producthunt_fetcher import ProductHuntFetcher
from ingestion.reddit_fetcher import RedditFetcher, parse_reddit_listing
from ingestion.x_fetcher import XFetcher
from datetime import datetime, timezone


HN_ITEMS = {
    1: {
        "id": 1,
        "type": "story",
        "title": "Show HN: Tiny CRM",
        "text": "<p>I track leads in a notebook.</p>",
        "by": "ada",
        "score": 10,
        "descendants": 1,
        "kids": [9],
        "url": "https://example.com/crm",
    },
    2: {
        "id": 2,
        "type": "story",
        "title": "A quiet essay about keyboards",
        "by": "bea",
        "score": 3,
        "descendants": 1,
    },
    3: {
        "id": 3,
        "type": "story",
        "title": "Database outage postmortem",
        "by": "cy",
        "score": 100,
        "descendants": 80,
    },
    9: {"id": 9, "type": "comment", "text": "I would pay for the reminder loop."},
}

REDDIT = {
    "data": {
        "children": [
            {"data": {"stickied": True, "id": "stick", "title": "Weekly thread", "permalink": "/r/SaaS/stick"}},
            {
                "data": {
                    "id": "abc",
                    "title": "I need a simpler invoicing loop",
                    "selftext": "Spreadsheets are failing me.",
                    "permalink": "/r/SaaS/comments/abc/invoicing/",
                    "author": "mina",
                    "score": 20,
                    "num_comments": 5,
                    "url": "https://www.reddit.com/r/SaaS/comments/abc/invoicing/",
                    "subreddit": "SaaS",
                }
            },
            {"data": {"id": "gone", "title": "[deleted]", "selftext": "[removed]", "permalink": "/r/SaaS/gone"}},
        ]
    }
}

FEED = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Acme &amp; invoices</title>
      <link>https://www.producthunt.com/posts/acme</link>
      <guid>https://www.producthunt.com/posts/acme</guid>
      <description><![CDATA[<p>Hello &amp; welcome. Chase unpaid invoices.</p>]]></description>
    </item>
  </channel>
</rss>
"""

TWEET = """<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <item>
      <title>Looking for a tiny invoicing bot</title>
      <link>https://nitter.example/ada/status/99</link>
      <guid>https://nitter.example/ada/status/99</guid>
      <description>Freelancers still chase invoices by hand.</description>
    </item>
  </channel>
</rss>
"""


REDDIT_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Software as a Service</title>
  <link rel="self" href="https://www.reddit.com/r/SaaS/.rss" />
  <entry>
    <author><name>/u/mina</name><uri>https://www.reddit.com/user/mina</uri></author>
    <content type="html">&lt;p&gt;Spreadsheets are failing me.&lt;/p&gt;</content>
    <id>t3_abc</id>
    <link href="https://www.reddit.com/r/SaaS/comments/abc/invoicing/" />
    <title>I need a simpler invoicing loop</title>
  </entry>
</feed>
"""


class FetcherTests(unittest.IsolatedAsyncioTestCase):
    def test_focus_rule_and_reddit_parser(self) -> None:
        self.assertEqual(dedupe_ids([[1, 2, "x"], [2, 4], None, [3]], 10), [1, 2, 4, 3])
        self.assertEqual(dedupe_ids([[1, 2, 3, 4]], 2), [1, 2])
        show = normalize_story(HN_ITEMS[1])
        assert show is not None
        self.assertEqual(show["id"], "hn_1")
        self.assertEqual(show["source"], "HackerNews")
        self.assertEqual(show["text"], "I track leads in a notebook.")
        self.assertIsNone(normalize_story({"id": 9, "deleted": True, "title": "gone"}))
        posts = parse_reddit_listing(REDDIT, "SaaS")
        self.assertEqual([post.external_id for post in posts], ["abc"])
        self.assertEqual(posts[0].author, "mina")

    def test_feed_parser_strips_html(self) -> None:
        posts = parse_feed(FEED, source="producthunt")
        self.assertEqual(posts[0].title, "Acme & invoices")
        self.assertIn("Hello & welcome", posts[0].body)
        self.assertNotIn("<p>", posts[0].body)

    async def test_hacker_news_keeps_show_hn_and_busy_threads(self) -> None:
        attempts = {"4": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/showstories.json"):
                return httpx.Response(200, json=[1, 2])
            if path.endswith("/askstories.json"):
                return httpx.Response(200, json=[2, 4])
            if path.endswith("/newstories.json"):
                return httpx.Response(200, json=[3, 5])
            item_id = path.rstrip("/").split("/")[-1].replace(".json", "")
            if item_id == "4":
                attempts["4"] += 1
                if attempts["4"] == 1:
                    return httpx.Response(429, headers={"Retry-After": "0"}, json={"retry": True})
            if item_id == "5":
                return httpx.Response(200, json=None)
            payload = {
                **HN_ITEMS.get(int(item_id), {}),
                "time": 1_700_000_000,
            }
            if item_id == "3":
                payload = {"id": 3, "deleted": True, "type": "story", "title": "gone"}
            if item_id == "2":
                payload = {
                    "id": 2,
                    "type": "story",
                    "title": "Ask HN: Who owns the keyboard",
                    "by": "bea",
                    "score": 3,
                    "descendants": 1,
                    "url": "https://example.com/keys",
                    "time": 1_700_000_100,
                }
            if item_id == "4":
                payload = {
                    "id": 4,
                    "type": "story",
                    "title": "Show HN: Invoice ping",
                    "by": "dee",
                    "score": 8,
                    "descendants": 2,
                    "time": 1_700_000_200,
                }
            return httpx.Response(200, json=payload)

        settings = Settings(hn_story_limit=10)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            errors: list[str] = []
            stories = await fetch_hn_stories(client, limit_per_feed=10, errors=errors)
            batch = await HackerNewsFetcher(settings, client).fetch()
        self.assertEqual([story["id"] for story in stories], ["hn_1", "hn_2", "hn_4"])
        self.assertEqual(stories[0]["text"], "I track leads in a notebook.")
        self.assertEqual(stories[1]["text"], "https://example.com/keys")
        self.assertEqual(stories[1]["url"], "https://news.ycombinator.com/item?id=2")
        self.assertEqual(stories[0]["timestamp"], 1_700_000_000)
        self.assertEqual(stories[0]["comments_count"], 1)
        self.assertGreaterEqual(attempts["4"], 2)
        self.assertEqual([post.external_id for post in batch.posts], ["hn_1", "hn_2", "hn_4"])
        self.assertFalse(any(post.external_id == "hn_3" for post in batch.posts))

    async def test_reddit_product_hunt_and_x_reads(self) -> None:
        def reddit_handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith(".rss"):
                return httpx.Response(200, text=REDDIT_RSS)
            return httpx.Response(403, text="blocked")

        settings = Settings(reddit_subs=["SaaS"], reddit_limit=10)
        async with httpx.AsyncClient(transport=httpx.MockTransport(reddit_handler)) as client:
            batch = await RedditFetcher(settings, client, min_interval=0).fetch()
        self.assertEqual(len(batch.posts), 1)
        self.assertEqual(batch.posts[0].external_id, "t3_abc")
        self.assertEqual(batch.posts[0].author, "/u/mina")
        self.assertIn("Spreadsheets are failing me", batch.posts[0].body)
        self.assertFalse(batch.errors)

        def ph_handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=FEED)

        async with httpx.AsyncClient(transport=httpx.MockTransport(ph_handler)) as client:
            batch = await ProductHuntFetcher(settings, client).fetch()
        self.assertEqual(batch.posts[0].source, "producthunt")
        self.assertIn("Acme", batch.posts[0].title)

        def broken_handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="this is not a feed")

        async with httpx.AsyncClient(transport=httpx.MockTransport(broken_handler)) as client:
            batch = await ProductHuntFetcher(settings, client).fetch()
        self.assertEqual(batch.posts, [])
        self.assertTrue(batch.errors)

        def x_handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "dead.example":
                return httpx.Response(503, text="down")
            return httpx.Response(200, text=TWEET)

        x_settings = Settings(
            nitter_instances=["https://dead.example", "https://live.example"],
            x_max_accounts_per_scan=1,
            x_posts_per_account=2,
            x_hashtags_per_scan=0,
        )
        moment = datetime(2026, 10, 27, 3, 0, tzinfo=timezone.utc)
        async with httpx.AsyncClient(transport=httpx.MockTransport(x_handler)) as client:
            fetcher = XFetcher(
                x_settings,
                client,
                accounts=["levelsio"],
                hashtags=[],
                clock=lambda: moment,
                min_interval=0,
            )
            batch = await fetcher.fetch()
        self.assertEqual(len(batch.posts), 1)
        self.assertEqual(batch.posts[0].external_id, "99")
        self.assertEqual(batch.posts[0].author, "levelsio")
        self.assertIn("https://dead.example", fetcher._down)
        self.assertEqual(batch.errors, [])


if __name__ == "__main__":
    unittest.main()
