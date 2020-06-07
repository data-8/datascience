test = {
  'name': 'Question 1.2.2',
  'points': 1,
  'suites': [
    {
      'cases': [
        {
          'code': r"""
          >>> from collections import Counter
          >>> g = train_lyrics.column('Genre')
          >>> r = np.where(test_lyrics['Title'] == "Grandpa Got Runned Over By A John Deere")[0][0]
          >>> t = test_20.row(r)
          >>> grandpa_expected_genre = Counter(np.take(g, np.argsort(fast_distances(t, train_20))[:9])).most_common(1)[0][0]
          >>> grandpa_genre == grandpa_expected_genre
          True
          """,
          'hidden': False,
          'locked': False
        },
      ],
      'scored': True,
      'setup': '',
      'teardown': '',
      'type': 'doctest'
    }
  ]
}
