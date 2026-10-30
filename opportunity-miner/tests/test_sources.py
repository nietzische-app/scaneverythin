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
from ingestion.funding_fetcher import FundingFetcher, collapse_funding, extract_funding, funding_post
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
        self.assertEqual(Settings().github_topics, [])

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
            quiet = await GitHubFetcher(Settings(), client).fetch()
        self.assertEqual(quiet.posts, [])
        self.assertEqual(quiet.errors, [])
        self.assertEqual(
            sources,
            ["hn", "reddit", "producthunt", "x", "github", "saas", "trends", "x-direct", "funding"],
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

    def test_funding_headline_keeps_company_round_and_amount(self) -> None:
        parsed = extract_funding(
            "Acme raises $12M Series A",
            "Acme helps customs brokers file declarations.",
        )
        self.assertEqual(parsed, ("Acme", "Series A", "$12M"))
        self.assertIsNone(extract_funding("How to write a deck", "Tips for founders."))
        self.assertIsNone(extract_funding("How founders closed their seed round", ""))
        turkish = extract_funding("Getir, Seri A turunda 50 milyon dolar yatırım aldı", "")
        self.assertEqual(turkish, ("Getir", "Seri A", "50 milyon"))
        post = funding_post(
            RawPost(
                source="rss",
                external_id="guid-1",
                url="https://techcrunch.com/2026/10/30/acme",
                title="Exclusive: Acme raises $12M Series A - TechCrunch",
                body="Acme helps customs brokers file declarations.",
            )
        )
        assert post is not None
        self.assertEqual(post.source, "funding")
        self.assertEqual(post.external_id, "acme:series-a")
        self.assertEqual(post.author, "techcrunch")
        self.assertIn("Signal: funding", post.body)
        self.assertIn("Company: Acme", post.body)
        self.assertIn("Round: Series A", post.body)
        self.assertIn("Amount: $12M", post.body)
        self.assertIn("What: Acme helps customs brokers file declarations.", post.body)

    def test_same_round_collapses_and_prefers_the_outlet_url(self) -> None:
        google = funding_post(
            RawPost(
                source="rss",
                external_id="g",
                url="https://news.google.com/rss/articles/acme",
                title="Acme raises $12M Series A - TechCrunch",
                body="Same round.",
            )
        )
        outlet = funding_post(
            RawPost(
                source="rss",
                external_id="t",
                url="https://techcrunch.com/2026/10/30/acme",
                title="Acme raises $12 million Series A",
                body="Same round from the outlet.",
            )
        )
        assert google is not None and outlet is not None
        self.assertEqual(google.external_id, outlet.external_id)
        self.assertEqual(google.content_hash, outlet.content_hash)
        kept = collapse_funding([google, outlet])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].url, "https://techcrunch.com/2026/10/30/acme")
        series_b = funding_post(
            RawPost(
                source="rss",
                external_id="b",
                url="https://techcrunch.com/2026/10/30/acme-b",
                title="Acme raises $40M Series B",
                body="Later round.",
            )
        )
        assert series_b is not None
        self.assertEqual(len(collapse_funding([google, series_b])), 2)

    async def test_a_dead_funding_feed_does_not_drop_the_others(self) -> None:
        feed = """<?xml version="1.0"?>
        <rss><channel>
          <item>
            <title>Acme raises $12M Series A</title>
            <link>https://www.eu-startups.com/2026/10/acme</link>
            <description>Acme files customs declarations for brokers.</description>
          </item>
          <item>
            <title>How to write a deck</title>
            <link>https://www.eu-startups.com/2026/10/deck</link>
            <description>Tips for founders raising a round.</description>
          </item>
        </channel></rss>"""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "www.eu-startups.com":
                return httpx.Response(200, text=feed)
            return httpx.Response(404, text="down")

        settings = Settings(
            funding_feeds=[
                "https://webrazzi.com/feed/",
                "https://www.eu-startups.com/feed/",
            ],
            funding_item_limit=8,
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            batch = await FundingFetcher(settings, client).fetch()
        self.assertEqual(len(batch.posts), 1)
        self.assertEqual(batch.posts[0].author, "eu-startups")
        self.assertEqual(batch.posts[0].external_id, "acme:series-a")
        self.assertTrue(batch.errors)
        self.assertIn("webrazzi", batch.errors[0])
        hosts = [item.split("/")[2] for item in Settings().funding_feeds]
        self.assertEqual(
            hosts,
            ["news.google.com", "webrazzi.com", "www.eu-startups.com", "techcrunch.com"],
        )


if __name__ == "__main__":
    unittest.main()
