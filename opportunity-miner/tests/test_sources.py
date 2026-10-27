"""Parsers and mocked reads for the four added sources."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from urllib.parse import unquote

import httpx

from tests.support import ROOT  # noqa: F401

from config import Settings
from ingestion import build_fetchers
from ingestion.github_fetcher import GitHubFetcher, select_repos
from ingestion.saas_pioneer_fetcher import (
    annotate_listing,
    business_model,
    flippa_posts,
    listing_post,
    newest_startup_urls,
    revenue_tier,
    SaasPioneerFetcher,
)
from ingestion.trends_fetcher import TrendsFetcher, rising_queries
from ingestion.x_direct_fetcher import XDirectFetcher, direct_query, posts_from_search
from database.models import RawPost


NOW = datetime(2026, 10, 27, tzinfo=timezone.utc)

SITEMAP = """<?xml version="1.0"?>
<urlset>
  <url><loc>https://app.acquire.com/startup/old-shop</loc><lastmod>2026-01-01</lastmod></url>
  <url><loc>https://app.acquire.com/startup/abc-ai-invoicing-8k-mrr</loc><lastmod>2026-10-20</lastmod></url>
</urlset>
"""

ACQUIRE_PAGE = """
<html><head>
<title>ignored</title>
<meta property="og:title" content="AI invoicing for freelancers">
<meta property="og:description" content="Freelancers chase unpaid invoices. $8k MRR subscription SaaS.">
</head></html>
"""

FLIPPA = """
<a href="https://flippa.com/12857417">Profitable B2B SaaS for coaches</a>
Monthly Revenue USD $19.1K
subscription SaaS
<a href="https://flippa.com/10">Tiny newsletter</a>
"""

IH = """<?xml version="1.0"?>
<rss version="2.0"><channel><item>
  <title>Hit $4,200 MRR with a reminder bot</title>
  <link>https://www.indiehackers.com/post/reminder</link>
  <guid>reminder</guid>
  <description>Agencies still chase invoices by hand. Subscription SaaS.</description>
</item></channel></rss>
"""


class SourceTests(unittest.IsolatedAsyncioTestCase):
    def test_github_keeps_only_fast_growing_repos(self) -> None:
        fresh = (NOW - timedelta(days=10)).strftime("%Y-%m-%dT00:00:00Z")
        old = (NOW - timedelta(days=1000)).strftime("%Y-%m-%dT00:00:00Z")
        posts = select_repos(
            [
                {
                    "id": 1,
                    "full_name": "acme/mcp",
                    "html_url": "https://github.com/acme/mcp",
                    "description": "MCP server that chases invoices",
                    "stargazers_count": 80,
                    "created_at": fresh,
                    "topics": ["mcp-server"],
                    "language": "Python",
                    "owner": {"login": "ada"},
                },
                {
                    "id": 2,
                    "full_name": "old/slow",
                    "html_url": "https://github.com/old/slow",
                    "stargazers_count": 500,
                    "created_at": old,
                },
                {
                    "id": 3,
                    "full_name": "new/tiny",
                    "html_url": "https://github.com/new/tiny",
                    "stargazers_count": 10,
                    "created_at": fresh,
                },
            ],
            now=NOW,
            min_stars=30,
            min_per_day=3,
        )
        self.assertEqual([post.external_id for post in posts], ["1"])
        self.assertIn("MCP server that chases invoices", posts[0].body)
        self.assertIn("/day", posts[0].body)
        self.assertEqual(posts[0].author, "ada")
        self.assertIn("mcp-server", Settings().github_topics)

    def test_listing_fields_and_flippa_cards(self) -> None:
        urls = newest_startup_urls(SITEMAP, 1)
        self.assertEqual(urls, ["https://app.acquire.com/startup/abc-ai-invoicing-8k-mrr"])
        post = listing_post("acquire", urls[0], ACQUIRE_PAGE, fallback_title="fallback")
        self.assertEqual(post.title, "AI invoicing for freelancers")
        self.assertIn("Problem / mechanism:", post.body)
        self.assertIn("Revenue tier: $1k–$10k MRR", post.body)
        self.assertIn("Business model: subscription SaaS", post.body)
        self.assertEqual(revenue_tier("no money here"), "unknown")
        self.assertEqual(business_model("a marketplace for leads"), "marketplace")
        cards = flippa_posts(FLIPPA, 5)
        self.assertEqual(cards[0].source, "flippa")
        self.assertEqual(cards[0].url, "https://flippa.com/12857417")
        self.assertIn("$10k+ MRR", cards[0].body)
        indie = annotate_listing(
            RawPost(
                source="indiehackers",
                external_id="reminder",
                url="https://www.indiehackers.com/post/reminder",
                title="Hit $4,200 MRR with a reminder bot",
                body="Agencies still chase invoices. Subscription SaaS.",
            ),
            "indiehackers",
        )
        self.assertIn("Revenue tier: $1k–$10k MRR", indie.body)
        self.assertEqual(indie.external_id, "reminder")

    def test_trends_keeps_breakout_and_200_percent(self) -> None:
        rows = rising_queries(
            {
                "default": {
                    "rankedList": [
                        {
                            "rankedKeyword": [
                                {"query": "mcp invoice server", "formattedValue": "+250%", "value": 250},
                                {"query": "pdf summarizer", "formattedValue": "+20%", "value": 20},
                                {"query": "agent guardrails", "formattedValue": "Breakout", "value": 5000},
                            ]
                        }
                    ]
                }
            },
            min_growth=200,
        )
        self.assertEqual([query for query, _label in rows], ["mcp invoice server", "agent guardrails"])

    def test_direct_query_and_tweets(self) -> None:
        query = direct_query(["#buildinpublic", "#saas", "mcp server", "ai agent SaaS"])
        self.assertIn("#buildinpublic", query)
        self.assertIn('"mcp server"', query)
        self.assertIn("-is:retweet", query)
        posts = posts_from_search(
            {
                "data": [
                    {
                        "id": "55",
                        "text": "Shipping an mcp server for invoices #buildinpublic",
                        "author_id": "9",
                        "public_metrics": {"like_count": 12, "reply_count": 3},
                    }
                ]
            },
            ["#buildinpublic", "mcp server"],
        )
        self.assertEqual(posts[0].url, "https://x.com/i/status/55")
        self.assertEqual(posts[0].external_id, "55")
        self.assertIn("Query: #buildinpublic", posts[0].body)
        self.assertEqual(posts[0].score, 12)

    async def test_fetchers_are_wired_and_read_fixtures(self) -> None:
        async with httpx.AsyncClient() as client:
            sources = [fetcher.source for fetcher in build_fetchers(Settings(), client)]
        self.assertEqual(
            sources,
            ["hn", "reddit", "producthunt", "x", "github", "saas", "trends", "x-direct"],
        )

        fresh = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%dT00:00:00Z")

        def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            if host == "api.github.com":
                query = unquote(request.url.query.decode())
                self.assertIn("topic:mcp-server", query)
                self.assertIn("topic:guardrails", query)
                return httpx.Response(
                    200,
                    json={
                        "items": [
                            {
                                "id": 7,
                                "full_name": "acme/guard",
                                "html_url": "https://github.com/acme/guard",
                                "description": "Guardrails for agents",
                                "stargazers_count": 40,
                                "created_at": fresh,
                                "topics": ["guardrails"],
                                "owner": {"login": "ada"},
                            }
                        ]
                    },
                )
            if "sitemap" in request.url.path:
                return httpx.Response(200, text=SITEMAP)
            if request.url.path.startswith("/startup/"):
                return httpx.Response(200, text=ACQUIRE_PAGE)
            if host == "www.indiehackers.com":
                return httpx.Response(200, text=IH)
            if host == "flippa.com":
                return httpx.Response(200, text=FLIPPA)
            if "explore" in request.url.path:
                body = {
                    "widgets": [
                        {"id": "RELATED_QUERIES", "token": "tok", "request": {"keyword": "AI tools"}}
                    ]
                }
                return httpx.Response(200, text=")]}'\n" + json.dumps(body))
            if "relatedsearches" in request.url.path:
                body = {
                    "default": {
                        "rankedList": [
                            {
                                "rankedKeyword": [
                                    {"query": "agent eval harness", "formattedValue": "+220%", "value": 220}
                                ]
                            }
                        ]
                    }
                }
                return httpx.Response(200, text=")]}'\n" + json.dumps(body))
            if host == "api.twitter.com":
                self.assertTrue(request.headers.get("authorization", "").startswith("Bearer test-token"))
                self.assertIn("mcp", unquote(request.url.query.decode()))
                return httpx.Response(
                    200,
                    json={"data": [{"id": "55", "text": "mcp server for saas #buildinpublic", "public_metrics": {}}]},
                )
            return httpx.Response(404, text="missing")

        settings = Settings(
            github_topics=["mcp-server", "guardrails"],
            github_min_stars=30,
            github_min_stars_per_day=3,
            acquire_sitemap_url="https://app.acquire.com/sitemap.xml",
            indiehackers_feed_url="https://www.indiehackers.com/feed.rss",
            flippa_url="https://flippa.com/buy/saas",
            saas_listing_limit=3,
            trends_keywords=["AI tools"],
            trends_min_growth=200,
            x_bearer_token="test-token",
            x_direct_queries=["#buildinpublic", "mcp server"],
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            github = await GitHubFetcher(settings, client).fetch()
            saas = await SaasPioneerFetcher(settings, client).fetch()
            trends = await TrendsFetcher(settings, client).fetch()
            direct = await XDirectFetcher(settings, client).fetch()
            missing = await XDirectFetcher(Settings(x_bearer_token=""), client).fetch()
        self.assertEqual(github.posts[0].url, "https://github.com/acme/guard")
        self.assertFalse(github.errors)
        sources = {post.source for post in saas.posts}
        self.assertEqual(sources, {"acquire", "indiehackers", "flippa"})
        self.assertFalse(saas.errors)
        self.assertIn("$8k MRR", saas.posts[0].body)
        self.assertEqual(trends.posts[0].title, "Search spike: agent eval harness")
        self.assertIn("+220%", trends.posts[0].body)
        self.assertEqual(direct.posts[0].external_id, "55")
        self.assertEqual(missing.posts, [])
        self.assertIn("X_BEARER_TOKEN is missing", missing.errors[0])


if __name__ == "__main__":
    unittest.main()
