import React from "react";

class SearchBar extends React.Component {
  state = { term: "machine learning", way: "tfidf" };

  onFormSubmit = event => {
    event.preventDefault();
    this.props.onFormSubmit(this.state.term, this.state.way);
  };

  render() {
    return (
      <div className="ui segment">
        <form className="ui form" onSubmit={this.onFormSubmit}>
          <div className="ui fluid action input left icon">
            <i className="search icon"></i>
            <textarea
              type="text"
              placeholder="Search..."
              onChange={e => this.setState({ term: e.target.value })}
              rows="4"
              style={{padding: '1em 2.5em'}}
            >{this.state.term}</textarea>
            <div>
              <select
                className="ui selection dropdown"
                onChange={e => this.setState({ way: e.target.value })}
              >
                <option value="tfidf">TF-IDF</option>
                <option value="bm25">BM25</option>
                <option value="glove">Glove</option>
              </select>
              <button
                className="fluid ui primary basic right labeled icon button"
                type="submit"
                style={{ padding: "2em 2.5em" }}
              >
                <i class="right arrow icon"></i>
                search
              </button>
            </div>
          </div>
        </form>
      </div>
    );
  }
}

export default SearchBar;
