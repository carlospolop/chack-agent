import os

from chack_tools.config import ToolsConfig
from chack_tools.forumscout_search import ForumScoutTool
from chack_tools.scientific_search import ScientificSearchTool
from chack_tools.social_network_agent import SocialNetworkAgentTool


class _Resp:
    status_code = 200
    text = "{}"

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_google_forums_and_news_send_rich_params(monkeypatch):
    os.environ["SERPAPI_API_KEY"] = "key"
    monkeypatch.setenv("SERPAPI_EXHAUSTION_CHECK_ENABLED", "0")
    seen = []

    def fake_get(url, params=None, timeout=20):
        seen.append(dict(params or {}))
        if params["engine"] == "google_forums":
            return _Resp(
                {
                    "organic_results": [
                        {
                            "title": "Forum thread",
                            "link": "https://forum.example/t",
                            "source": "Forum",
                            "date": "Yesterday",
                            "snippet": "A discussion",
                        }
                    ],
                    "related_searches": [{"query": "related forum"}],
                }
            )
        return _Resp(
            {
                "news_results": [
                    {
                        "title": "News story",
                        "link": "https://news.example/s",
                        "source": "News",
                        "date": "Today",
                    }
                ]
            }
        )

    monkeypatch.setattr("chack_tools.forumscout_search.requests.get", fake_get)
    helper = ForumScoutTool(ToolsConfig(forumscout_max_results=3))

    forums = helper.search_google_forums(
        "topic",
        page=2,
        gl="US",
        hl="en",
        location="Austin, Texas, United States",
        tbs="qdr:w",
    )
    news = helper.search_google_news("", gl="US", hl="en", so=1, story_token="story")

    assert seen[0]["engine"] == "google_forums"
    assert seen[0]["start"] == 10
    assert "page" not in seen[0]
    assert seen[0]["gl"] == "us"
    assert seen[0]["hl"] == "en"
    assert seen[0]["location"] == "Austin, Texas, United States"
    assert seen[0]["tbs"] == "qdr:w"
    assert "related forum" in forums
    assert seen[1]["engine"] == "google_news"
    assert "q" not in seen[1]
    assert seen[1]["so"] == 1
    assert seen[1]["story_token"] == "story"
    assert "News story" in news


def test_google_news_rejects_invalid_query_token_combinations():
    helper = ForumScoutTool(ToolsConfig())

    assert (
        helper.search_google_news("topic", story_token="story")
        == "ERROR: Google News q cannot be used together with topic/publication/section/story/kgmid tokens"
    )
    assert (
        helper.search_google_news("", kgmid="/m/0vzm", topic_token="topic")
        == "ERROR: Google News kgmid must be used alone without other advanced tokens"
    )


def test_google_trends_and_trending_now_format_signal(monkeypatch):
    os.environ["SERPAPI_API_KEY"] = "key"
    monkeypatch.setenv("SERPAPI_EXHAUSTION_CHECK_ENABLED", "0")
    seen = []

    def fake_get(url, params=None, timeout=20):
        seen.append(dict(params or {}))
        if params["engine"] == "google_trends":
            return _Resp(
                {
                    "interest_over_time": {
                        "timeline_data": [
                            {"date": "Jan 2026", "values": [{"extracted_value": 42}]}
                        ]
                    },
                    "related_queries": {
                        "rising": [{"query": "topic controversy", "value": "+300%"}]
                    },
                }
            )
        return _Resp(
            {
                "trending_searches": [
                    {
                        "query": "breaking social trend",
                        "search_volume": "200K+",
                        "articles": [
                            {
                                "title": "Context article",
                                "link": "https://news.example/context",
                                "source": "News",
                            }
                        ],
                    }
                ]
            }
        )

    monkeypatch.setattr("chack_tools.forumscout_search.requests.get", fake_get)
    helper = ForumScoutTool(ToolsConfig(forumscout_max_results=5))

    trends = helper.search_google_trends(
        "topic",
        data_type="TIMESERIES",
        date="now 7-d",
        geo="US",
        gprop="youtube",
    )
    trending = helper.search_google_trends_trending_now(geo="US", hours=24, only_active=True)

    assert seen[0]["engine"] == "google_trends"
    assert seen[0]["gprop"] == "youtube"
    assert "Jan 2026: 42" in trends
    assert "topic controversy" in trends
    assert seen[1]["engine"] == "google_trends_trending_now"
    assert seen[1]["only_active"] == "true"
    assert "breaking social trend" in trending
    assert "Context article" in trending


def test_google_videos_and_social_profiles(monkeypatch):
    os.environ["SERPAPI_API_KEY"] = "key"
    monkeypatch.setenv("SERPAPI_EXHAUSTION_CHECK_ENABLED", "0")
    seen = []

    def fake_get(url, params=None, timeout=20):
        seen.append(dict(params or {}))
        if params["engine"] == "google_videos":
            return _Resp(
                {
                    "video_results": [
                        {
                            "title": "Video result",
                            "link": "https://video.example/v",
                            "source": "Video Site",
                            "date": "2 days ago",
                        }
                    ]
                }
            )
        if params["engine"] == "instagram_profile":
            return _Resp(
                {
                    "profile_results": {
                        "username": "example",
                        "link": "https://instagram.com/example",
                        "followers": "10K",
                        "posts": [
                            {
                                "caption": "post caption",
                                "link": "https://instagram.com/p/1",
                                "likes": "20",
                            }
                        ],
                    },
                    "serpapi_pagination": {"next_page_token": "next"},
                }
            )
        return _Resp(
            {
                "profile_results": {
                    "name": "Example Page",
                    "link": "https://facebook.com/example",
                    "followers": "2K",
                }
            }
        )

    monkeypatch.setattr("chack_tools.forumscout_search.requests.get", fake_get)
    helper = ForumScoutTool(ToolsConfig(forumscout_max_results=5))

    videos = helper.search_google_videos("topic", page=3, gl="US")
    instagram = helper.get_instagram_profile("@example", next_page_token="abc")
    facebook = helper.get_facebook_profile("example")

    assert seen[0]["engine"] == "google_videos"
    assert seen[0]["start"] == 20
    assert "Video result" in videos
    assert seen[1]["engine"] == "instagram_profile"
    assert seen[1]["profile_id"] == "example"
    assert seen[1]["next_page_token"] == "abc"
    assert "post caption" in instagram
    assert "Next page token: next" in instagram
    assert seen[2]["engine"] == "facebook_profile"
    assert "Example Page" in facebook


def test_youtube_search_details_and_full_transcript(monkeypatch):
    os.environ["SERPAPI_API_KEY"] = "key"
    monkeypatch.setenv("SERPAPI_EXHAUSTION_CHECK_ENABLED", "0")
    seen = []

    def fake_get(url, params=None, timeout=20):
        seen.append(dict(params or {}))
        if params["engine"] == "youtube":
            return _Resp(
                {
                    "video_results": [
                        {
                            "title": "Important video",
                            "link": "https://www.youtube.com/watch?v=abc123XYZ",
                            "channel": {"name": "Channel"},
                            "published_date": "1 day ago",
                            "views": "50K views",
                        }
                    ],
                    "serpapi_pagination": {"next_page_token": "NEXTYT"},
                    "related_searches": [{"query": "related video query"}],
                }
            )
        if params["engine"] == "youtube_video":
            return _Resp(
                {
                    "video_result": {
                        "title": "Important video",
                        "link": "https://www.youtube.com/watch?v=abc123XYZ",
                        "description": "Long description",
                        "views": "50K",
                    },
                    "comments": [{"author": "user", "text": "source comment", "likes": "5"}],
                    "related_videos": [{"title": "Related", "link": "https://youtu.be/rel987"}],
                }
            )
        return _Resp(
            {
                "transcript": [
                    {"start": "0:00", "text": "segment one"},
                    {"start": "0:05", "text": "segment two"},
                    {"start": "0:10", "text": "segment three"},
                    {"start": "0:15", "text": "segment four"},
                ]
            }
        )

    monkeypatch.setattr("chack_tools.scientific_search.requests.get", fake_get)
    helper = ScientificSearchTool(ToolsConfig(scientific_max_results=5))

    search = helper.search_youtube_videos("topic")
    details = helper.get_youtube_video_details("https://www.youtube.com/watch?v=abc123XYZ")
    transcript = helper.get_youtube_video_transcript("abc123XYZ")
    capped = helper.get_youtube_video_transcript("abc123XYZ", max_segments=2)

    assert seen[0]["engine"] == "youtube"
    assert "video_id: abc123XYZ" in search
    assert "Next page token: NEXTYT" in search
    assert "related video query" in search
    assert seen[1]["engine"] == "youtube_video"
    assert seen[1]["v"] == "abc123XYZ"
    assert "source comment" in details
    assert "Related" in details
    assert "all 4 segments" in transcript
    assert "segment four" in transcript
    assert "Artifact JSON:" in transcript
    assert "Artifact text:" in transcript
    assert "top 2 of 4 segments" in capped
    assert "segment three" not in capped


def test_social_agent_includes_enriched_social_tools():
    helper = SocialNetworkAgentTool(
        ToolsConfig(),
        model_name="gpt-test",
        model_provider="openai",
    )

    names = {getattr(tool, "name", "") for tool in helper._build_subagent_tools()}

    assert "search_google_trends" in names
    assert "search_google_trends_trending_now" in names
    assert "search_google_videos" in names
    assert "get_instagram_profile" in names
    assert "get_facebook_profile" in names
    assert "get_youtube_video_details" in names
